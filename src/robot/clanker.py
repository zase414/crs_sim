from ctu_mitsubishi import Rv6s
import numpy as np

class Clanker(Rv6s):
    def _send_and_receive(self, msg: str) -> str:
        self._connection.write(msg.encode("utf-8"))
        res = self._connection.read_until(b"\r").decode("utf-8")
        if not res.endswith("\r"):
            raise TimeoutError(f"No complete reply to {msg!r}, got {res!r}")
        return res

    def move_rel(self, offset_pose):
        qi = self.get_q()
        curr_pose = self.fk(qi)

        target_pose = curr_pose @ offset_pose

        self.move_abs(self, target_pose)

    def move_abs(self, target_pose):
        qi = self.get_q()

        # IK + only solutions within joint limits
        qs = [q for q in self.ik(target_pose) if self.in_limits(q)]
        if not qs:
            print("Skipping move: unreachable, target", target_pose[:3, 3].round(3))
            return False

        q = min(qs, key=lambda q: np.linalg.norm(q - qi))

        self.move_to_q(q)
        self.wait_for_motion_stop()
