import matplotlib.pyplot as plt
import os 

def create_figures(train_metrics, test_metrics, model_name):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    train_loss = extract_metric(train_metrics['train'], 'net_loss')
    val_loss = extract_metric(train_metrics['val'], 'net_loss')
    test_loss = test_metrics['net_loss']

    train_accuracy = extract_metric(train_metrics['train'], 'net_accuracy')
    val_accuracy = extract_metric(train_metrics['val'], 'net_accuracy')
    test_accuracy = test_metrics['net_accuracy']

    train_precision = extract_metric(train_metrics['train'], 'precision')
    val_precision = extract_metric(train_metrics['val'], 'precision')
    test_precision = test_metrics['precision']

    train_recall = extract_metric(train_metrics['train'], 'recall')
    val_recall = extract_metric(train_metrics['val'], 'recall')
    test_recall = test_metrics['recall']
    
    
    print(train_precision, train_recall)
    train_f1 = [2 * p * r / max(1, (p + r)) for p, r in zip(train_precision, train_recall)]
    val_f1 = [2 * p * r / max(1, (p + r))  for p, r in zip(val_precision, val_recall)]
    test_f1 = (2 * test_precision * test_recall) / max(1, test_precision + test_recall)



    plt.suptitle(model_name)

    axs[0].plot(train_loss, label='Train', c='blue')

    axs[0].plot(val_loss, label='Validation', c='orange')
    axs[0].axhline(y=test_loss, color='red', linestyle='--', label='Test')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot accuracy curves
    axs[1].plot(train_accuracy, label='Train', c='blue')
    axs[1].plot(val_accuracy, label='Validation', c='orange')
    axs[1].axhline(y=test_accuracy, color='red', linestyle='--', label='Test')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()


    axs[2].plot(train_precision, label='Train Precision')
    axs[2].plot(val_precision, label='Validation Precision')
    axs[2].plot(train_recall, label='Train Recall')
    axs[2].plot(val_recall, label='Validation Recall')
    axs[2].plot(train_f1, label='Train F1')
    axs[2].plot(val_f1, label='Validation F1')
    axs[2].axhline(y=test_precision, color='red', linestyle='--', label='Test Precision')
    axs[2].axhline(y=test_recall, color='green', linestyle='--', label='Test Recall')
    axs[2].axhline(y=test_f1, color='blue', linestyle='--', label='Test F1')
    axs[2].set_title('Precision, Recall, and F1 Curve')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Score')
    axs[2].legend()

    try:
        os.makedirs(os.path.join('results', 'figures', model_name))
    except:
        print()

    plt.savefig(os.path.join('results', 'figures', model_name, 'out.png')) 
    
    return None

def extract_metric(metric_list, key):
    return [item[key] for item in metric_list]


# def accuracy_histogram(model_name, plot_name, category_metrics):
    
#     fig, ax = plt.subplots(1, len(category_metrics.keys()), figsize=(15, 5))

#     metrics = ['tp', 'fp', 'tn', 'fn']

#     for i, c in enumerate(category_metrics.keys()):
#         ax[i].bar(metrics, category_metrics[c])
#         ax[i].set_title(c)
#         ax[i].set_yscale('log')

#     fig.suptitle(plot_name + "-" + model_name)

#     plt.savefig(os.path.join('results', 'figures', f'{plot_name + "-" + model_name}.png')) 
#     plt.show()
    



def category_histograms(model_name, plot_name, category_metrics):
    
    fig, ax = plt.subplots(1, len(category_metrics.keys()), figsize=(15, 5))

    metrics = ['tp', 'fp', 'tn', 'fn']

    for i, c in enumerate(category_metrics.keys()):
        ax[i].bar(metrics, category_metrics[c])
        ax[i].set_title(c)
        ax[i].set_yscale('log')

    fig.suptitle(plot_name + "-" + model_name)

    plt.savefig(os.path.join('results', 'figures', f'{plot_name + "-" + model_name}.png')) 
    plt.show()
    

