import matplotlib.pyplot as plt

def plot_target_classes(df):
    num_bins = len(df['tag_target_class'].unique())

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(df['tag_target_class'], num_bins)

    ax.set_xlabel('Target class')
    ax.set_ylabel('Articles')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()