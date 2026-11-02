from ctu_mitsubishi import Rv6s
import numpy as np
import time

class Clanker(Rv6s):
    #: no new byte for this long after a "\r" means the reply is complete [s]
    REPLY_GAP = 0.02

    def _send_and_receive(self, msg: str) -> str:
        # replies end with "\r", "\r\n" or several "\r" on one row: read up to a
        # "\r", then take whatever else arrives in the same burst
        self._connection.write(msg.encode("utf-8"))
        res = b""
        while not res.strip(b"\r\n"):  # skip empty rows / a "\n" left over
            chunk = self._connection.read_until(b"\r")
            if not chunk.endswith(b"\r"):
                raise TimeoutError(f"No complete reply to {msg!r}, got {res + chunk!r}")
            res += chunk
            while not res.endswith(b"\n"):
                time.sleep(self.REPLY_GAP)
                n = self._connection.in_waiting
                if not n:
                    break
                res += self._connection.read(n)
        return res.decode("utf-8").strip("\r\n") + "\r"

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
