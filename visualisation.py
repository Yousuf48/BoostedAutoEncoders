from common import utils, plt


def show_images(images, title=None):
    grid = utils.make_grid(images, nrow=8, padding=3)
    plt.imshow(grid.permute(1, 2, 0))

    plt.axis("off")
    if title:
        plt.title(title)

    plt.show()
