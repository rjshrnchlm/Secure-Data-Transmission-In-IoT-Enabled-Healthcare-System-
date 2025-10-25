import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn.metrics import roc_curve

def plot_Results():
    for a in range(1):
        Eval = np.load('Evaluate_all.npy', allow_pickle=True)[a]

        Terms = ['Execution Time(S)', 'Memory Size(KB) ', 'Security(%)', 'Trust(%)', 'Cost', 'Average Latency', 'Energy Consumption']
        for b in range(len(Terms)):
            learnper = [1, 2, 3, 4, 5]

            # X = np.arange(5)
            # plt.plot(learnper, Eval[:, 0, b], color='#aaff32', linewidth=3, marker='o', markerfacecolor='#aaff32', markersize=14,
            #          label="OOA-DSR")
            # plt.plot(learnper, Eval[:, 1, b], color='#ad03de', linewidth=3, marker='o', markerfacecolor='#ad03de', markersize=14,
            #          label="MAO-DSR")
            # plt.plot(learnper, Eval[:, 2, b], color='#8c564b', linewidth=3, marker='o', markerfacecolor='#8c564b', markersize=14,
            #          label="SCO-DSR")
            # plt.plot(learnper, Eval[:, 3, b], color='#ff000d', linewidth=3, marker='o', markerfacecolor='#ff000d', markersize=14,
            #          label="CSO-DSR")
            # plt.plot(learnper, Eval[:, 4, b], color='k', linewidth=3, marker='o', markerfacecolor='k', markersize=14,
            #          label="RPCSO-DSR")
            #
            # labels = ['100Tx', '200Tx', '300Tx', '400Tx', '500Tx']
            # plt.xticks(learnper, labels)
            #
            # plt.xlabel('No of Transactions(Tx)')
            # plt.ylabel(Terms[b])
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            #            ncol=3, fancybox=True, shadow=True)
            # path1 = "./Results/Dataset_%s_%s_line.png" % (a + 1, Terms[b])
            # plt.savefig(path1)
            # plt.show()

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Eval[:, 5, b], color='#aaff32', width=0.10, label="20 Nodes")
            ax.bar(X + 0.10, Eval[:, 6, b], color='#ad03de', width=0.10, label="40 Nodes")
            ax.bar(X + 0.20, Eval[:, 7, b], color='#8c564b', width=0.10, label="60 Nodes")
            ax.bar(X + 0.30, Eval[:, 8, b], color='#ff000d', width=0.10, label="80-Nodes")
            ax.bar(X + 0.40, Eval[:, 9, b], color='k', width=0.10, label="100-Nodes")
            # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))

            labels = ['100Tx', '200Tx', '300Tx', '400Tx', '500Tx']
            plt.xticks(X + 0.20, labels)

            plt.xlabel('No of Transactions(Tx)')
            plt.ylabel(Terms[b])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar.png" % (a + 1, Terms[b])
            plt.savefig(path1)
            plt.show()

def plot_results_intrution():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC',
             'FOR', 'PT', 'CSI', "BA", 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Algorithm = ['TERMS', 'DO', 'WSO', 'DMO', 'DFA', 'Proposed']
    Classifier = ['TERMS', 'RNN', 'VGG16', 'CNN', 'MDARNN', 'PROPOSED']
    # value = eval[4, :, 4:]

    Activation_Function = ['Linear', 'Relu', 'Sigmoid', 'Tanh', 'Softmax']
    for n in range(2):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[2], eval.shape[1]))
            for k in range(eval.shape[2]):
                # for l in range(eval.shape[1]):
                if j == 9:
                    Graph[k, :] = eval[n, :, k, Graph_Term[j] + 4]
                else:
                    Graph[k, :] = eval[n, :, k, Graph_Term[j] + 4]

            plt.plot(Activation_Function, Graph[0, :], color='y', linewidth=5, marker='D', markerfacecolor='y',
                     markersize=12,
                     label="Linear")
            plt.plot(Activation_Function, Graph[1, :], color=[0.5, 0.9, 0.9], linewidth=5, marker='D', markerfacecolor=[0.5, 0.9, 0.9],
                     markersize=12,
                     label="Relu")
            plt.plot(Activation_Function, Graph[2, :], color='b', linewidth=5, marker='D', markerfacecolor='b',
                     markersize=12,
                     label="Sigmoid")
            plt.plot(Activation_Function, Graph[3, :], color=[0.7, 0.7, 0.9], linewidth=5, marker='D', markerfacecolor=[0.7, 0.7, 0.9],
                     markersize=12,
                     label="Tanh")
            plt.plot(Activation_Function, Graph[4, :], color='k', linewidth=5, marker='D', markerfacecolor='k',
                     markersize=12,
                     label="Softmax")
            plt.xticks(Activation_Function, ('LEA-SA-OTESN', 'AOA-SA-OTESN', 'FLO-SA-OTESN', 'MFO-SA-OTESN', 'EMFO-SA-OTESN'), rotation=10)
            # plt.xlabel('Heuristic Algorithm')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='best')
            path1 = "./Results/Dataset_%s_%s_line.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[5, :], color=[0.5, 0.5, 0.9], width=0.15, label="Linear")
            ax.bar(X + 0.15, Graph[6, :], color=[0.5, 0.9, 0.9], width=0.15, label="Relu")
            ax.bar(X + 0.30, Graph[7, :], color=[0.7, 0.7, 0.9], width=0.15, label="Sigmoid")
            ax.bar(X + 0.45, Graph[8, :], color='y', width=0.15, label="Tanh")
            ax.bar(X + 0.60, Graph[9, :], color='k', width=0.15, label="Softmax")
            plt.xticks(X + 0.10, ('BiLSTM', 'SA-BiLSTM', 'XGBoost', 'TESN', 'EMFO-SA-OTESN'))
            plt.grid(axis='y')
            # plt.xlabel('Classifier')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc=1)
            path1 = "./Results/Dataset_%s_%s_BarGraph.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()




def plot_tables():
    # matplotlib.use('TkAgg')
    eval = np.load('Evaluate_all1.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC',
             'FOR', 'PT', 'CSI', "BA", 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    Algorithm = ['TERMS', 'DMO-ODR-AM', 'DFA-ODR-AM', 'EOO-ODR-AM', 'GAO-ODR-AM', 'ARGAO-ODR-AM']
    Classifier = ['TERMS', '1DCNN', 'BiTrans', 'LSTM', 'DR-AM', 'ARGAO-ODR-AM']
    Epoch = ['50', '100', '150', '200', '250']
    for m in range(eval.shape[0]):
        for n in range(eval.shape[1]):
            value = eval[m, n, :, 4:11]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Terms[0:7])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, :])
            print('--------------------------------------------------Dataset - ', m + 1, ' -', Epoch[n],
                  ' -Algorithm Comparison',
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[0:7])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
            print('---------------------------------------------------Dataset - ', m + 1, ' -', Epoch[n],
                  ' -Classifier Comparison',
                  '--------------------------------------------------')
            print(Table)



def Plot_ROC():
    lw = 2
    cls = ['BiLSTM', 'SA-BiLSTM', 'XGBoost', 'TESN', 'EMFO-SA-OTESN']
    for a in range(1):
        Actual = np.load('Target_1.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "r", "crimson", "gold", "black"])  # "cornflowerblue","darkorange", "aqua"
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

def plot_Fitness():
    for a in range(2):
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['LEA-SA-OTESN', 'AOA-SA-OTESN', 'FLO-SA-OTESN', 'MFO-SA-OTESN', 'EMFO-SA-OTESN']

        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('-------------------------------------------------- Dataset_'+str(a+1)+'Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='#7ebd01', linewidth=3, marker='>', markerfacecolor='blue', markersize=12,
                 label="LEA-SA-OTESN")
        plt.plot(iteration, conv[1, :], color='#ef4026', linewidth=3, marker='>', markerfacecolor='red', markersize=12,
                 label="AOA-SA-OTESN")
        plt.plot(iteration, conv[2, :], color='#12e193', linewidth=3, marker='>', markerfacecolor='green', markersize=12,
                 label="FLO-SA-OTESN")
        plt.plot(iteration, conv[3, :], color='#ff0490', linewidth=3, marker='>', markerfacecolor='yellow',
                 markersize=12,
                 label="MFO-SA-OTESN")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=12,
                 label="EMFO-SA-OTESN")
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/convergence_%s.jpg"  %(str(a+1))
        plt.savefig(path1)
        plt.show()

# plot_Fitness()
# Plot_ROC()
# plot_tables()
plot_Results()
# plot_results_intrution()
