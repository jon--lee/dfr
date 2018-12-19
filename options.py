


class Options():

    def __init__(self):
        self.iters = None
        self.trials = None

    def copy(self):
        opt = Options()
        attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for attr in attributes:
            value = getattr(self, attr)
            setattr(opt, attr, value)
        return opt

    def load_config(self):
        return


    def load_dict(self, d):
        for key in d.keys():
            value = args[key]
            setattr(self, key, value)

    def load_args(self, args):
        args = vars(args)
        for key in args:
            value = args[key]
            setattr(self, key, value)



if __name__ == '__main__':
    opt = Options()
    opt2 = opt.copy()

