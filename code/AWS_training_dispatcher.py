from AWS_training_utils import *
import logging

'''
Pass params to training in the form:
    [learning_rate,
     batch_size,
     num_epochs,
     steps_per_epoch,
     validation_steps,
     workers,
     optimizer]

    Optimizer is either 'Adam' or 'Nadam' (only learning rate is specified)

'''

params1 = [0.01,
           64,
           15,
           500,
           50,
           2,
           'Adam']

params2 = [0.01,
           128,
           15,
           500,
           50,
           2,
           'Adam']

params3 = [0.01,
           256,
           15,
           500,
           50,
           2,
           'Adam']

params4 = [0.01,
           256,
           15,
           500,
           50,
           2,
           'Nadam'
           ]

params_list = [params1, params2, params3, params4]

def get_weights_file_name(params):
    name = 'weights_{0}_rate_{1}_batch_{2}_epochs_{3}_epoch_steps_{4}_valid_steps_{5}_opt'.format(str(params[0][2:]),
                                                                                                  params[1],
                                                                                                  params[2],
                                                                                                  params[3],
                                                                                                  params[4],
                                                                                                  params[6]
                                                                                                  )

def return_params(params):
    params_string = ['Learning Rate: ', 'Batches: ', 'Num Epochs: ', 'Steps per Epoch: ', 'Validation Steps: ', 'Workers: ', 'Optimizer: ']
    out_list = [p + str(params[i]) for i,p in enumerate(params_string)]
    return ', '.join(out_list)

def get_time():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

def main(params_list):

    for p in params:
        logging.info('Start time: {}'.format(datetime.datetime.now()))
        logging.info('Parameters: {}'.format(return_params(p)))
        logging.info('Training model')

        weights_file_name = get_weights_file_name(p)

        logging.info('Save weights to filename: {}'.format(weights_file_name))

        train_model(*p)

        logging.info('Finished model training at {}'.format(datetime.datetime.now()))



if __name__ == '__main__':
    logging.basicConfig(filename='{}.log'.format(get_time()),level=logging.INFO)

    try:
        main()
    except Exception as e:
        logging.critical(e)
        logging.critical('Stopped execution at {}'.format(datetime.datetime.now()))
        raise RuntimeError('Failed during execution.')









