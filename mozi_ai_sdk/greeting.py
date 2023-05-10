def greeting(name: str, excited: bool = False) -> str:
    message = "Hello, {}".format(name)
    if excited:
        message += "!!!"
    return message


class A:
    def f(self) -> int:  # Type of self inferred (A)
        return 2


class B(A):
    def f(self) -> int:
        return 3

    def g(self) -> int:
        return 4


def foo(a: A) -> None:
    print(a.f())  # 3
    # a.g()         # Error: "A" has no attribute "g"


foo(B())  # OK (B is a subclass of A)


from typing import Callable


def arbitrary_call(f: Callable[..., int]) -> int:
    return f("x") + f(y=2)  # OK


arbitrary_call(ord)  # No static error, but fails at runtime
arbitrary_call(open)  # Error: does not return an int
arbitrary_call(1)  # Error: 'int' is not callable
