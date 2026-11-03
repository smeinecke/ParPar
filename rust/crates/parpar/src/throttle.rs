//! Port of `lib/throttlequeue.js` - DelayThrottle / RateThrottle /
//! ReqRateThrottle / NoThrottle.
//!
//! The JS implementation is callback/timer based; the Rust pipeline is
//! synchronous, so `pass` blocks the calling thread until the request is
//! allowed through, returning a guard whose drop is the JS `readDone()`.
//! `cancel`/`flush` wake every queued request with `cancelled` set.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// JS `emptyFn` read-done callback - a no-op guard returned from `pass`.
pub struct ReadDone;

enum Kind {
    /// `DelayThrottle` - every request delayed by `wait_time` ms.
    Delay { wait_time: f64 },
    /// `RateThrottle` - `max_amount` units per `time_window` ms.
    Rate { max_amount: i64, time_window: i64 },
    /// `ReqRateThrottle` - RateThrottle with cost fixed at 1.
    ReqRate { max_amount: i64, time_window: i64 },
    /// `NoThrottle`.
    None,
}

struct State {
    /// queued waiters, woken FIFO on timeout/cancel
    queue: VecDeque<u64>,
    next_id: u64,
    /// RateThrottle debt bookkeeping
    debt: i64,
    debt_time: Option<Instant>,
    cancelled: bool,
}

pub struct ThrottleQueue {
    kind: Kind,
    state: Mutex<State>,
    cv: Condvar,
}

impl ThrottleQueue {
    /// JS `ThrottleQueue.NoThrottle`.
    pub fn none() -> Self {
        Self::new(Kind::None)
    }
    /// JS `new ThrottleQueue.DelayThrottle(time)` (ms).
    pub fn delay(wait_time: f64) -> Self {
        Self::new(Kind::Delay { wait_time })
    }
    /// JS `new ThrottleQueue.RateThrottle(maxAmount, timeWindow)` (ms).
    pub fn rate(max_amount: f64, time_window: f64) -> Self {
        Self::new(Kind::Rate {
            max_amount: max_amount as i64,
            time_window: time_window as i64,
        })
    }
    /// JS `new ThrottleQueue.ReqRateThrottle(maxAmount, timeWindow)`.
    pub fn req_rate(max_amount: f64, time_window: f64) -> Self {
        Self::new(Kind::ReqRate {
            max_amount: max_amount as i64,
            time_window: time_window as i64,
        })
    }

    fn new(kind: Kind) -> Self {
        ThrottleQueue {
            kind,
            state: Mutex::new(State {
                queue: VecDeque::new(),
                next_id: 0,
                debt: 0,
                debt_time: None,
                cancelled: false,
            }),
            cv: Condvar::new(),
        }
    }

    /// JS `_adjustDebt` - staggered debt decay; returns precise current debt.
    fn adjust_debt(&self, st: &mut State, (max_amount, time_window): (i64, i64)) -> f64 {
        if time_window <= 0 {
            return 0.0;
        }
        let now = Instant::now();
        if let Some(dt) = st.debt_time {
            let elapsed_ms = now.saturating_duration_since(dt).as_millis() as f64;
            let periods = (elapsed_ms / time_window as f64).floor();
            if periods > 0.0 {
                st.debt -= (periods * max_amount as f64) as i64;
                let dt2 = dt + Duration::from_millis((periods * time_window as f64) as u64);
                st.debt_time = Some(dt2);
                if st.debt <= 0 {
                    st.debt = 0;
                    st.debt_time = Some(now);
                }
            }
            let dt = st.debt_time.unwrap();
            let elapsed_ms = now.saturating_duration_since(dt).as_millis() as f64;
            st.debt as f64 - elapsed_ms * max_amount as f64 / time_window as f64
        } else {
            st.debt_time = Some(now);
            st.debt as f64
        }
    }

    /// JS `pass(cost, cb)` - blocks until admitted; `true` in the return =
    /// JS `cancelled`. Caller drops `ReadDone` after the read completes.
    pub fn pass(&self, cost: f64) -> (bool, ReadDone) {
        let st = self.state.lock().unwrap();
        match &self.kind {
            Kind::None => (false, ReadDone),
            Kind::Delay { wait_time } => {
                // JS: every request queued for waitTime; cancelled -> true
                if *wait_time > 0.0 && !st.cancelled {
                    let (st2, _t) = self
                        .cv
                        .wait_timeout(st, Duration::from_millis(*wait_time as u64))
                        .unwrap();
                    (st2.cancelled, ReadDone)
                } else {
                    (st.cancelled, ReadDone)
                }
            }
            Kind::Rate {
                max_amount,
                time_window,
            } => self.rate_pass(st, *max_amount, *time_window, cost),
            Kind::ReqRate {
                max_amount,
                time_window,
            } => self.rate_pass(st, *max_amount, *time_window, 1.0),
        }
    }

    fn rate_pass(
        &self,
        mut st: std::sync::MutexGuard<'_, State>,
        max_amount: i64,
        time_window: i64,
        cost: f64,
    ) -> (bool, ReadDone) {
        if time_window <= 0 {
            // JS: debt always 0 -> always admitted
            return (false, ReadDone);
        }
        loop {
            let current_debt = self.adjust_debt(&mut st, (max_amount, time_window));
            if current_debt < max_amount as f64 && st.queue.is_empty() && !st.cancelled {
                // up cost + let through
                st.debt += cost as i64;
                return (false, ReadDone);
            }
            if st.cancelled {
                return (true, ReadDone);
            }
            // queued - wait for the debt window (JS timer)
            let wait_ms = ((current_debt - max_amount as f64 + 1.0) * time_window as f64
                / max_amount as f64)
                .ceil()
                .max(0.0);
            st.next_id += 1;
            let id = st.next_id;
            st.queue.push_back(id);
            st = if wait_ms > 0.0 {
                self.cv
                    .wait_timeout(st, Duration::from_millis(wait_ms as u64 + 1))
                    .unwrap()
                    .0
            } else {
                self.cv.wait(st).unwrap()
            };
            // timed-out entries run (JS _timeout always runs queue head)
            if let Some(pos) = st.queue.iter().position(|&i| i == id) {
                st.queue.remove(pos);
                if st.cancelled {
                    return (true, ReadDone);
                }
                st.debt += cost as i64;
                return (false, ReadDone);
            }
            // removed by _finish
            if st.cancelled {
                return (true, ReadDone);
            }
        }
    }

    /// `createThrottle` - build from a validated spec (validation already
    /// done by `generate::create_throttle` in the solver).
    pub fn from_spec(spec: &crate::cliparse::ThrottleSpec) -> Option<Self> {
        use crate::cliparse::ThrottleMode;
        match spec.mode {
            ThrottleMode::Delay => Some(Self::delay(spec.time)),
            ThrottleMode::Rate => Some(Self::rate(spec.size, spec.time)),
            ThrottleMode::ReqRate => {
                Some(Self::req_rate(spec.count.parse().unwrap_or(0.0), spec.time))
            }
        }
    }

    /// JS `cancel()` - all queued requests get `cancelled=true`.
    pub fn cancel(&self) {
        let mut st = self.state.lock().unwrap();
        st.cancelled = true;
        st.queue.clear();
        st.debt = 0;
        st.debt_time = None;
        self.cv.notify_all();
    }

    /// JS `flush()` - all queued requests run with `cancelled=false`.
    pub fn flush(&self) {
        let mut st = self.state.lock().unwrap();
        st.queue.clear();
        st.debt = 0;
        st.debt_time = None;
        self.cv.notify_all();
    }
}
