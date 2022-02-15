
import matplotlib.pyplot as plt
import numpy as np

def v_loss(train_losses, val_losses):
    plt.plot(train_losses[:150], label="train loss")
    plt.plot(val_losses[:150], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")
    plt.savefig("loss.png")
    plt.show()

def v_latent(train_z, val_z, test_z): 
    plt.figure(figsize=(10, 10))
    plt.scatter(train_z[1:, 0], train_z[1:, 1], marker='.',  c = "black", label = "train")
    plt.scatter(val_z[1:, 0], val_z[1:, 1], marker='.',  c = "blue", label = "val")
    plt.scatter(test_z[1:, 0], test_z[1:, 1], marker='.',  c = "red", label = "test")
    plt.grid()
    plt.legend(loc="upper right")
    plt.savefig("2d_latent.png")
    plt.show()

def v_masquerade(user_id, test_each_loss):
    label = []
    with open("label.txt") as f:
        for _ in range(100):
            label.append(list(map(int, f.readline().split())))
    f.close()
    label = np.array(label).T
    pos_mas = np.where(label[user_id-1])[0]

    plt.plot(test_each_loss, label="test loss")
    plt.scatter(pos_mas, np.array(test_each_loss)[pos_mas], c = "orange", label = "masquerade")
    plt.legend()
    plt.savefig("User"+str(user_id) + "_loss_masquerade_position.png")
    plt.show()