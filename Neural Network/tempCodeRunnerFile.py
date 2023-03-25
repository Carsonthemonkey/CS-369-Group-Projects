    net = Network([2, 5, 1])
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    for _ in range(1000):
        for i, t in zip(inputs, targets):
            net.train(i, t)
    net.graph_mse()
    for i, t in zip(inputs, targets):
        self.assertAlmostEqual(t[0], net.predict(i)[0], delta=0.2)
