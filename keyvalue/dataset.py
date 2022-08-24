import torch
import torch.distributions as tdist

import matplotlib.pyplot as plt


# Centerpoint is (4, 4)
# the keypoint is (2, 2), (4, 2), (6, 2), (2, 4) (6, 4), (6, 2), (6, 4) , (6, 6)


def dataset_generator():
    a = tdist.Normal(torch.Tensor([2.0, 2.0]), torch.Tensor([0.25]))
    b = tdist.Normal(torch.Tensor([2.0, 4.0]), torch.Tensor([0.25]))
    c = tdist.Normal(torch.Tensor([2.0, 6.0]), torch.Tensor([0.25]))
    d = tdist.Normal(torch.Tensor([4.0, 2.0]), torch.Tensor([0.25]))
    e = tdist.Normal(torch.Tensor([4.0, 6.0]), torch.Tensor([0.25]))
    f = tdist.Normal(torch.Tensor([6.0, 2.0]), torch.Tensor([0.25]))
    g = tdist.Normal(torch.Tensor([6.0, 4.0]), torch.Tensor([0.25]))
    h = tdist.Normal(torch.Tensor([6.0, 6.0]), torch.Tensor([0.25]))


    class_a = a.sample((50,))
    class_b = b.sample((50,))
    class_c = c.sample((50,))
    class_d = d.sample((50,))
    class_e = e.sample((50,))
    class_f = f.sample((50,))
    class_g = g.sample((50,))
    class_h = h.sample((50,))
    group = [class_a, class_b, class_c, class_d, class_e, class_f, class_g, class_h]

    return group




if __name__ == '__main__':
    fig, ax = plt.subplots()
    group = dataset_generator()
    for i, classofg in enumerate(group):
        ax.scatter(classofg[:,0], classofg[:, 1], marker='o', label=str(i+1))
    plt.show()
    plt.savefig('dataset.png')
    print('\u2713 Dataset Generator - Checked')
