from os import walk



if __name__ == '__main__':
    filenames = next(walk('captchas_train'), (None, None, []))[
        2]  # [] if no file
    print(filenames)
    