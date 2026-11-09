from solution import models, pipelines


def names():
    return sorted(list(pipelines.PIPELINES) + list(models.LEARNED))


def is_learned(name):
    return name in models.LEARNED


def make(name):
    return models.LEARNED[name]()


def predict(name, match, train_runs=None):
    if not is_learned(name):
        return pipelines.run(name, match)
    return make(name).fit(train_runs).predict(match)
