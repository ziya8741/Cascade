with open("main.py", "a") as f:
    f.write('\nif __name__ == "__main__":\n    main()\n')
print("Guard appended.")
