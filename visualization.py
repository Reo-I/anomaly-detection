
import matplotlib.pyplot as plt

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