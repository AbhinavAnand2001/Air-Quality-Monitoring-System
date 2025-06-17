import time

def test_python_efficiency():
    """
    A simple Python function to test if Python is running and producing output efficiently.
    It performs a basic calculation and measures the execution time.
    """
    print("--- Python Efficiency Test ---")

    start_time = time.time()

    # Perform a simple, slightly time-consuming calculation
    result = 0
    for i in range(1, 10_000_001): # Loop 10 million times
        result += i * 2 / 3

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Calculation finished. Result: {result:.2f}")
    print(f"Execution time: {execution_time:.4f} seconds")

    if execution_time < 1.0:
        print("Python appears to be running quite efficiently for this task.")
    else:
        print("Execution time seems a bit high, but Python is running correctly.")

    print("--- Test Complete ---")

if __name__ == "__main__":
    test_python_efficiency()