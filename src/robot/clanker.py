from ctu_mitsubishi import Rv6s
import numpy as np
import time

class Clanker(Rv6s):
    #: time the controller program needs after RUNMoveit before it accepts targets [s]
    PROGRAM_START_DELAY = 1.5
    #: how long to wait for a commanded motion to start before resending it [s]
    MOTION_START_TIMEOUT = 1.0
    MOTION_RESENDS = 2

    def _send_and_receive(self, msg: str) -> str:
        # the controller terminates replies with "\r" only, readline() would wait
        # for the 1 s serial timeout on every command
        # drop stale replies so they do not shift every following reply by one
        self._connection.reset_input_buffer()
        self._connection.write(msg.encode("utf-8"))
        res = self._connection.read_until(b"\r").decode("utf-8")
        if not res.endswith("\r"):
            raise TimeoutError(f"No complete reply to {msg!r}, got {res!r}")
        return res

    def initialize(self):
        was_initialized = self._initialized
        super().initialize()
        if not was_initialized and hasattr(self, "_connection"):
            # targets sent while the Moveit program is still starting get lost
            time.sleep(self.PROGRAM_START_DELAY)

    def move_to_q(self, q):
        q = np.asarray(q, dtype=float)
        q_before = self.get_q()
        for attempt in range(self.MOTION_RESENDS + 1):
            super().move_to_q(q)
            # already there, or moving (this motion or one queued before it)
            start = time.time()
            while time.time() - start < self.MOTION_START_TIMEOUT:
                q_now = self.get_q()
                if (np.all(np.abs(q_now - q) < 0.0087)
                        or np.any(np.abs(q_now - q_before) > 0.002)):
                    return
                time.sleep(0.05)
            print(f"[WARN] robot did not start moving to {np.rad2deg(q).round(1)}, "
                  f"resending ({attempt + 1}/{self.MOTION_RESENDS})")
        print("[ERR] robot ignored the motion command")

    def wait_for_motion_stop(self, timeout: float = 30, poll_interval: float = 0.1,
                             tol: float = 0.0087) -> bool:
        stopped = super().wait_for_motion_stop(timeout, poll_interval, tol)
        if not stopped:
            print(f"[WARN] robot did not reach {np.rad2deg(self._last_q).round(1)} "
                  f"within {timeout} s")
        return stopped

    def move_rel(self, offset_pose):
        qi = self.get_q()
        curr_pose = self.fk(qi)

        target_pose = curr_pose @ offset_pose

        return self.move_abs(target_pose)

    def move_abs(self, target_pose):
        assert target_pose[2,3] > 0.10, "[ERR] TABLE DESTRUCTION AVOIDED"
        qi = self.get_q()

        # IK + only solutions within joint limits
        qs = [q for q in self.ik(target_pose) if self.in_limits(q)]
        if not qs:
            print("Skipping move: unreachable, target", target_pose[:3, 3].round(3))
            return False

        q = min(qs, key=lambda q: np.linalg.norm(q - qi))

        self.move_to_q(q)
        self.wait_for_motion_stop()
        return True
