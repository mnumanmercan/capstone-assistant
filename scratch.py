def make_counter(start):
    count = start
    def increment():
        # ??? burada bir sorun var, çalıştır gör
        nonlocal count
        count += 1
        return count
    return increment

c = make_counter(10)
print(c())