import math


def findFixedPoint(f, x0, maxIters=100, minDelta_x=1e-10):
    prev_x = None
    x = x0
    for i in range(maxIters):
        print(f"iter {i}:", x)
        x = f(x)
        if prev_x is not None and abs(x - prev_x) < minDelta_x:
            print("Stopped early due to small changes in x")
            break
        prev_x = x
    return x


if __name__ == "__main__":
    # solving for root of  e^x - x - 2 = 0 by solving the equivalent problem
    # e^x = x + 2
    # x = ln(x + 2)
    # so, need to find fixed point of ln(x + 2)
    x0 = 1.15
    findFixedPoint(lambda x: math.log(x + 2), x0)
