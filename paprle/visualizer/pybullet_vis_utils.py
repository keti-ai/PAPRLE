from contextlib import contextmanager
import os
import sys

@contextmanager
def stdout_redirected(to: str = os.devnull):
    """Temporarily redirects `sys.stdout` to the specified file

    This context manager is useful for silencing output or redirecting it to a
    file or other writable stream during the execution of a code block.

    Example:
        ```
        import os
        with stdout_redirected(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")
        ```

    Args:
        to (str): The target file where stdout should be redirected.
            Defaults to `os.devnull` (silencing output).
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
