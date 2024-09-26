'''
# compare our metric V1 and V2 to SSIM, PSNR, IS, and FID by training models from scratch on synthetic, real, synth+real...etc
'''

import train, test, utils
import os
import sys

ds_names = ["car_accidents", "cifar", "birds"]
metrics_names = ["SSIM", "PSNR", "IS", "FID", "Ours"]
is_hec = "/home/kerim/PycharmProjects/SemanticSegmentation" not in os.getcwd()

num_epochs = 20# should be 20 for all exps
# metric_id = 3# SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
isFnT = True

def normal_mode():
    ds_id = 2  # 0 (car acidents), 1 (cifar), 2 (birds)
    ds_percs = [1.0]#[0.01, 0.2, 0.5, 0.9, 1.0]
    f = open('expInfo_ds_{}.txt'.format(ds_names[ds_id]), 'a')
    # Your normal program logic here
    print("Running in normal mode")
    for metric_id in range(4, 5): # SSIM (0), PSNR (1),IS (2), FID (3), Ours (4)
        for ds_perc in ds_percs:  # by default 1.0 (all dataset); it is only for pure synth or pure real
            for net_id in range(0, 6):  # 0 (AlexNet), 1 (EffNet), 2 (ViT), 3 (SwinTransformer), 4 (VGG), 5 (REGNet). 30 (myCustom Model)
                for train_mode in range(5, 6):  # -1 (Synth nonphoto), 0 (Syn), 1 (Rel), 2 (Syn+Rel), 3(Syn+ + Rel), 4(Syn + Rel+), 5(Syn*0.25 + Rel0.5),
                                                # 6(Syn*0.5 + Rel0.5), 20 (Synth nonphoto), 21 (Synth nonphoto + real)

                    print("net_id: {}, ds_name: {}, metric_names: {}, train_mode: {}, ds_perc: {}".format(net_id, ds_names[ds_id],
                                                                                                                metrics_names[metric_id],
                                                                                                                train_mode,
                                                                                                                ds_perc))

                    if is_hec:
                        model_path2save = "/storage/hpc/01/kerim/GANsPaper/Checkpoints/{}/BestModel.pth".format(
                            ds_names[ds_id])
                    else:
                        model_path2save = "Checkpoints/{}/BestModel.pth".format(ds_names[ds_id])

                    # let's train our model for n_epochs
                    model,best_model, best_accuracy, test_loader, train_losses, val_accuracies = train.train(num_epochs,
                                                                                                             ds_names[ ds_id],
                                                                                                             metrics_names[metric_id],
                                                                                                             net_id,
                                                                                                             train_mode,
                                                                                                             is_hec,
                                                                                                             model_path2save,
                                                                                                             ds_perc,isFnT)

                    print('Finished Training')
                    print('The best validation accuracy was %d %%' % best_accuracy)
                    # utils.plot_info(train_losses, val_accuracies)

                    # let's test our model on our real dataset <<I should check the @bug there>>
                    # best_model = torch.load("Checkpoints/cifar/BestModel_Corr.pth")
                    ground_truth, predictions, test_accuracy = test.test(model_path2save, best_model, test_loader, ds_names[ds_id])
                    # plot confusion matrix of the test set
                    # utils.plot_confusion_matrix(ground_truth, predictions)
                    f.write("net_id: {}, ds_name: {}, metric_names: {}, train_mode: {}, ds_perc: {}, fnt: {}, numEpch: {}, test_accuracy: {} \n".format(net_id,
                                                                                                                ds_names[ds_id],
                                                                                                                metrics_names[metric_id],
                                                                                                                train_mode,
                                                                                                                ds_perc,isFnT,num_epochs,
                                                                                                                test_accuracy))
                    f.flush()


def parameter_mode(args):
    # f = open('expInfo.txt', 'a')
    # ds_percs:  0.01 0.1 0.2 0.5 0.9 1
    # net_ids: 0 (AlexNet), 1 (EffNet), 2 (ViT), 3 (SwinTransformer) 4 (VGG), 5 (REGNet). 30 (myCustom Model)
    # train_modes:  -1 (Synth nonphoto), 0 (Syn), 1 (Rel), 2 (Syn+Rel), 3(Syn+ + Rel), 4(Syn + Rel+), 5(Syn* + Rel),
    # metric_id = 0  # SSIM (0), PSNR (1), IS (2), FID (3), Ours (4)
    # ds_id: 0 (car accidents), 1 (cifar), 2 (birds)
    # python A01_Main_allExps.py ds_percs  -- net_ids -- train_modes -- metrics_ids -- ds_id
    # python A01_Main_allExps.py 0.01 0.1 0.2 0.5 0.9 1 -- 0 1 2 3 4 5 -- -1 0 1 2 --  0 1 2 3 4 -- 0
    # Your logic for processing command-line arguments here
    print("Running in parameter mode")
    print("Arguments:", args)
    # Convert the command-line arguments to arrays
    separator_indices = [i for i, arg in enumerate(args) if arg == "--"]

    ds_percs = list(map(float, args[:separator_indices[0]]))
    net_ids = list(map(int, args[separator_indices[0] + 1:separator_indices[1]]))
    train_modes = list(map(int, args[separator_indices[1] + 1:separator_indices[2]]))
    metric_ids = list(map(int, args[separator_indices[2] + 1:separator_indices[3]]))
    ds_id = int(args[separator_indices[3] + 1])

    print("Running in parameter mode")
    print("Received ds percentage:", ds_percs)
    print("Received new ids:", net_ids)
    print("Received train modes:", train_modes)
    print("Received metric ids:", metric_ids)
    print("Received integer:", ds_id)

    for metric_id in metric_ids: # SSIM (0), PSNR (1),IS (2), FID (3), Ours (4)
        for ds_perc in ds_percs:  # by default 1.0 (all dataset); it is only for pure synth or pure real
            f = open('expInfo_ds_{}.txt'.format(ds_names[ds_id]), 'a')
            for net_id in net_ids:  # 0 (AlexNet), 1 (EffNet), 2 (ViT), 3 (SwinTransformer) 4 (VGG), 5 (REGNet). 30 (myCustom Model)
                for train_mode in train_modes: # -1 (Synth nonphoto), 0 (Syn), 1 (Rel), 2 (Syn+Rel), 3(Syn+ + Rel), 4(Syn + Rel+), 5(Syn*0.25 + Rel0.5), 6(Syn*0.5 + Rel0.5)
                    # 20 (Synth nonphoto), 21 (Synth nonphoto + real)

                    if is_hec:
                        model_path2save = "/storage/hpc/01/kerim/GANsPaper/Checkpoints/{}/BestModel.pth".format(ds_names[ds_id])
                    else:
                        model_path2save = "Checkpoints/{}/BestModel.pth".format(ds_names[ds_id])

                    print("net_id: {}, ds_name: {}, metric_names: {}, train_mode: {}, ds_perc: {}".format(net_id, ds_names[ds_id],
                                                                                                                metrics_names[metric_id],
                                                                                                                train_mode,
                                                                                                                ds_perc))


                    model,best_model, best_accuracy, test_loader, train_losses, val_accuracies = train.train(num_epochs,
                                                                                                             ds_names[ds_id],
                                                                                                             metrics_names[metric_id],
                                                                                                             net_id,
                                                                                                             train_mode,
                                                                                                             is_hec,
                                                                                                             model_path2save,
                                                                                                             ds_perc,isFnT)

                    print('Finished Training')
                    print('The best validation accuracy was %d %%' % best_accuracy)
                    # utils.plot_info(train_losses, val_accuracies)

                    # let's test our model on our real dataset
                    ground_truth, predictions, test_accuracy = test.test(model_path2save, best_model, test_loader, ds_names[ds_id])

                    f.write("net_id: {}, ds_name: {}, metric_names: {}, train_mode: {}, ds_perc: {}, fnt: {}, num_epochs: {}, test_accuracy: {} \n".format(net_id,
                                                                                                                ds_names[ds_id],
                                                                                                                metrics_names[metric_id],
                                                                                                                train_mode,
                                                                                                                ds_perc,isFnT,num_epochs,
                                                                                                                test_accuracy))
                    f.flush()


def main():
    if len(sys.argv) > 1:
        parameter_mode(sys.argv[1:])
    else:
        normal_mode()


if __name__ == "__main__":
    main()
