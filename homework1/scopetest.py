u = [1, 1, 1, 1]


def changeU():
    u[1] = 0


def wrap():
    changeU()


if __name__ == "__main__":
    print(u)
    wrap()
    print(u)
