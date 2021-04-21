
--model_file <path to the trained model> --normalization [ bn | in | bin | ln | gn | nn | inbuilt ] 
--n [ 1 |  2 | 3  ] --test_data_file <path to a csv with each line representing an image> 
--output_file <file containing the prediction in the same order as in the input csv>


# plot_errors(train_loss, valid_loss)
# plot_percentile(percentile_1, percentile_20, percentile_80, percentile_99)
# save_print("train",logger)
# a,b = test_model(model, train_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# save_print("val",logger)
# a,b = test_model(model, val_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# save_print("test",logger)
# a,b = test_model(model, test_loader)
# micro = get_microF1(a,b)
# macro = get_macroF1(a,b)
# save_print(f"Macro f1:{macro}, Micro f1:{micro}", logger)
# with open(FOLDER + "/train_loss.txt", "wb") as fp:   #Pickling
#    pickle.dump(train_loss, fp)
# with open(FOLDER + "/val_loss.txt", "wb") as fp:   #Pickling
#    pickle.dump(valid_loss, fp)
# logger.close()