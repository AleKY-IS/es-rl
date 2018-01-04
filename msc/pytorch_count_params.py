def pytorch_count_params(model):
    "count number trainable parameters in a pytorch model"
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params