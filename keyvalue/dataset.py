import torch
from torch.nn.functional import one_hot
import torch.distributions as tdist
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt


# Centerpoint is (4, 4)
# the keypoint is (2, 2), (4, 2), (6, 2), (2, 4) (6, 4), (6, 2), (6, 4) , (6, 6)


def dataset_generator():
    a = tdist.Normal(torch.Tensor([2.0, 2.0]), torch.Tensor([0.3]))
    b = tdist.Normal(torch.Tensor([6.0, 4.0]), torch.Tensor([0.3]))
    c = tdist.Normal(torch.Tensor([4.0, 6.0]), torch.Tensor([0.3]))
    d = tdist.Normal(torch.Tensor([6.0, 2.0]), torch.Tensor([0.3]))
    e = tdist.Normal(torch.Tensor([2.0, 4.0]), torch.Tensor([0.3]))
    f = tdist.Normal(torch.Tensor([6.0, 6.0]), torch.Tensor([0.3]))
    g = tdist.Normal(torch.Tensor([2.0, 6.0]), torch.Tensor([0.3]))
    h = tdist.Normal(torch.Tensor([4.0, 2.0]), torch.Tensor([0.3]))


    class_a = a.sample((50,))
    label_a = torch.full((50,), 0, dtype=torch.long)
    class_b = b.sample((50,))
    label_b = torch.full((50,), 1, dtype=torch.long)
    class_c = c.sample((50,))
    label_c = torch.full((50,), 2, dtype=torch.long)
    class_d = d.sample((50,))
    label_d = torch.full((50,), 3, dtype=torch.long)
    class_e = e.sample((50,))
    label_e = torch.full((50,), 4, dtype=torch.long)
    class_f = f.sample((50,))
    label_f = torch.full((50,), 5, dtype=torch.long)
    class_g = g.sample((50,))
    label_g = torch.full((50,), 6, dtype=torch.long)
    class_h = h.sample((50,))
    label_h = torch.full((50,), 7, dtype=torch.long)
    inp = torch.cat([class_a, class_b, class_c, class_d, class_e, class_f, class_g, class_h], dim=0)
    label = torch.cat([label_a, label_b, label_c, label_d, label_e, label_f, label_g, label_h], dim=0)
    group = [class_a, class_b, class_c, class_d, class_e, class_f, class_g, class_h]
    return inp, label, group


class clustered2D(Dataset):
    def __init__(self):
        
        inp, label, group = dataset_generator()
        fig, ax = plt.subplots()
        for i, classofg in enumerate(group):
            ax.scatter(classofg[:, 0], classofg[:, 1], marker='o', label=str(i+1))
            plt.show()
            plt.savefig('../images/dataset.png')
        print('\u2713 Dataset Generator - Checked')
        self.x_data = inp
        self.y_data = label
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

    





        
