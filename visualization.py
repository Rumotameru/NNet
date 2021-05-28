import os
import matplotlib.pyplot as plt
import seaborn as sns


class_names = ["Normal", "ALL"]


def class_balance(df):
    ax = sns.countplot(x='labels', data=df, palette=['#E2485A', '#679B8B'])
    for p in ax.patches:
        ax.annotate('{:d}'.format(p.get_height()), (p.get_x() + 0.05, 300), fontsize = 26, fontweight = 100, color = '#ffffff')
        ax.annotate('{:.2f}%'.format(p.get_height()/len(df)*100), (p.get_x() + 0.05, 1300), fontsize = 10, fontweight = 700, color = '#ffffff')
    plt.show()


def preview(sample, path):
    cr_dir = os.getcwd()
    plt.figure(figsize=(16, 9))
    for i in range(len(sample)):
        ax = plt.subplot(5, 10, i + 1)
        img = plt.imread(os.path.join(cr_dir, path, sample['images'].iloc[i]))
        plt.imshow(img)
        lbl = sample['labels'].iloc[i]
        plt.axis("off")
        plt.title(class_names[lbl])
    plt.show()


def print_losses(train_l, valid_l):
    plt.plot(train_l, label='Training loss')
    plt.plot(valid_l, label='Validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.show()


def print_acc(v_acc, t_acc):
    plt.plot(v_acc, label='Valid')
    plt.plot(t_acc, label='Test')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(frameon=False)
    plt.show()


