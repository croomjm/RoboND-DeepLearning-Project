from AWS_training_utils import *
import logging
import time
import datetime

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
           20,
           400,
           50,
           2,
           'Nadam']

params2 = [0.005,
           64,
           20,
           400,
           50,
           2,
           'Nadam']

params3 = [0.002,
           64,
           25,
           400,
           50,
           2,
           'Nadam']

params_list = [params1, params2, params3]

def get_weights_file_name(params):
    name = 'weights_{0}_rate_{1}_batch_{2}_epochs_{3}_epoch_steps_{4}_valid_steps_{5}_opt_{6}'.format(str(params[0])[2:],
                                                                                                  params[1],
                                                                                                  params[2],
                                                                                                  params[3],
                                                                                                  params[4],
                                                                                                  params[6],
                                                                                                  get_time()
                                                                                                  )

    return name

def return_params(params):
    params_string = ['Learning Rate: ', 'Batches: ', 'Num Epochs: ', 'Steps per Epoch: ', 'Validation Steps: ', 'Workers: ', 'Optimizer: ']
    out_list = [p + str(params[i]) for i,p in enumerate(params_string)]
    return ', '.join(out_list)

def get_time():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr

def main(params_list):

    for p in params_list:
        logging.info('Start time: {}'.format(datetime.datetime.now()))
        logging.info('Parameters: {}'.format(return_params(p)))
        logging.info('Training model')

        weights_file_name = get_weights_file_name(p)

        logging.info('Save weights to filename: {}'.format(weights_file_name))

        p.append(weights_file_name)

        train_model(*p)

        logging.info('Finished model training at {}'.format(datetime.datetime.now()))



if __name__ == '__main__':
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='{}.log'.format(get_time()),
                    filemode='w')
    #logging.basicConfig(filename='{}.log'.format(get_time()),level=logging.INFO)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    try:
        main(params_list)
    except Exception as e:
        logging.critical(e)
        logging.critical('Stopped execution at {}'.format(datetime.datetime.now()))
        raise RuntimeError('Failed during execution.')









