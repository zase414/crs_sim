from ctu_mitsubishi import Rv6s

class Clanker(Rv6s):
    def _send_and_receive(self, msg: str) -> str:
        self._connection.write(msg.encode("utf-8"))
        res = self._connection.read_until(b"\r").decode("utf-8")
        if not res.endswith("\r"):
            raise TimeoutError(f"No complete reply to {msg!r}, got {res!r}")
        return res