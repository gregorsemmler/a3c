import signal
import time
import multiprocessing


class GracefulExit(Exception):
    pass


def signal_handler(signum, frame):
    print("Received exit signal")
    raise GracefulExit()


def subprocess_function():
    try:
        sem = multiprocessing.Semaphore()
        print("Acquiring semaphore")
        sem.acquire()
        print("Semaphore acquired")

        print("Blocking on semaphore - waiting for SIGTERM")
        sem.acquire()
    except GracefulExit:
        print("Subprocess exiting gracefully")


def subprocess2(idx):
    while True:
        print(f"Subprocess {idx}")
        time.sleep(10.0)


def main():
    num_processes = 10

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # # Start a subprocess and wait for it to terminate.
    # p = multiprocessing.Process(target=subprocess2, args=())
    # p.start()
    #
    # print(f"Subprocess pid: {p.pid}")
    # p.join()

    processes = []

    for idx in range(num_processes):
        p = multiprocessing.Process(target=subprocess2, args=(idx,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pass


if __name__ == "__main__":
    main()
