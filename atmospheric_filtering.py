import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, join, Column, MaskedColumn
from astropy.io.ascii import read, write
from math import sqrt
import time as time
from tqdm import tqdm
import os
import sys
from matplotlib.patches import Rectangle

cwd = os.getcwd()
fileE , fileW, Ffile, alldata, file_E_W = '57856 East.csv','57856 West.csv','Frame.csv', '57856.csv','E_W_merged.csv'
Mb = 20 #chunk_size for fast reader, in bytes


def list_to_array(list_of_lists):
    #convers a list of lists to a list of numpy arrays
    list_of_arrays = [np.array(alist) for alist in list_of_lists]
    return list_of_arrays


def flatten_arrays(list_of_arrays):

    #flattens a list of n dimensional arrays to a list 1 dimensional arrays
    list_of_flattens = [arr.flatten() for arr in list_of_arrays]
    return list_of_flattens


def get_kstars2(frames, rng, target, flagged, max_ensemble_size):

    '''
    the function filters any stars that are not within the ensemble range, or are in the list 'flagged'
    :param frame: The set of stars with their corresponding mags, MagErrs, and coordinates
    :param rng: The distance from the target star in pixels
    :param target: string of the target star's name
    :param flagged: a list of stars that have been removed from the ensemble
    :return: a table of stars that are all within the given ensemble range
    '''

    # reording data by 'StarName'
    frames = frames.group_by('StarName')
    keys = frames.groups.keys
    mask = frames.groups.keys['StarName'] == target
    target_data = frames.groups[mask]
    all_data = frames.groups[mask]
    targ_x, targ_y = target_data[0]['X'], target_data[0]['Y']
    targ_mag = target_data[0]['MAG']
    #outfile = os.path.join(cwd, 'Data', 'Outputs', 'lightcurves', 'test.csv')
    if flagged is None:
        flagged = []
    ens_size = 0
    for i in tqdm(range(len(keys)), desc = 'Getting stars in range: '):

        key = keys[i]['StarName']
        key = keys['StarName'] == key

        if key in flagged:
            print('Star found, but in flagged')
            continue

        star_data = frames.groups[key]
        ens_mag = star_data[0]['MAG']
        ens_x, ens_y = star_data[0]['X'], star_data[0]['Y']


        if ens_x == targ_x and ens_y == targ_y:
            #print('Star found, but target')
            continue

        '''if abs(ens_mag - targ_mag) > 2:
            print('Star found, but wrong mag')
            continue'''

        if sqrt((ens_x - targ_x)*(ens_x - targ_x) + (ens_y - targ_y)*(ens_y - targ_y)) <= rng:
            all_data = vstack([all_data, star_data])
            #ens_size += 1
            #print("Star found, stacking..")
        #if ens_size >= max_ensemble_size:
            #print('Done')
           # break
    #write(all_data, outfile, overwrite= True)
    return all_data


def get_mj(frame):

    '''
    :param:   frame: The set of stars with their corresponding mags, MagErrs, and coordinates
    :return: the weighted ensemble average 'mj' of the magnitude for the current frame (see eclipse mapping paper) and the correspoonding error
    '''

    merr = np.array(frame['MagError'])
    mag = np.array(frame['MAG'])
    wk = list(map(lambda x: 1/(x*x), merr))
    sum_mw = sum([m*w for m, w in zip(mag, wk)])
    mj = sum_mw/sum(wk)
    merr_sqrd = [err*err for err in merr]
    merr_ens = 1/sqrt(sum([1/err for err in merr_sqrd]))
    return mj, merr_ens


def get_enesemble_data(frames, list_of_keys, dtypes = None):
    '''
    :param list_of_keys: a set of star names
    :param frames: The data set containing the time series for all observed stars with the Star Name, X and Y corrdinates, magrror,
            and mag
    :return: a data set of stars with the corresponding star name keys
    '''

    if dtypes is None:
        dtypes = ('S10', 'f8','f8','f8','f8','f8','f8' )
    keys = frames.groups.keys
    new_data = Table(names = frames.colnames, dtype = dtypes)

    for key in list_of_keys:
        key = keys['StarName'] == key
        star_data = frames.groups[key]
        #combining star_data with the whole dataset
        new_data = vstack([new_data, star_data])

    return new_data


def get_sigmas(ensemble_data):
    '''
    :param ensemble_data: A dataset with all stars in the ensemble
    :return: the standard deviations of their lightcurves
    '''

    sigmas = []
    names = []
    ensemble_data = ensemble_data.group_by('StarName')

    keys = ensemble_data.groups.keys
    key_names = ensemble_data.groups.keys['StarName']
    for i in range(len(keys)):
        key = keys[i]['StarName']
        key = keys['StarName'] == key
        star_data = ensemble_data.groups[key]
        sigma = np.std(star_data['MAG'])
        sigmas.append(sigma)
        name = key_names[i]
        names.append(name)
    sig_dict = dict(zip(names, sigmas))
    return sig_dict


def preprocess(frames, target, flagged, ens_dists):
    '''
    :param frames: The data set containing the time series for all observed stars with the Star Name, X and Y corrdinates, magrror,
            and mag
    :param target: a string of a StarName in the dataset
    :return: a table with a unified time series for the get_kstars function.
            Unified meaning that all the times for all the stars have the same set of modified julian dates.
            If a stars time series doesn't match, it gets removed from the data set. Also returns a dictionary called 'ensemble_sigs' withe names and sigmas for each star in the ensemble
    '''

    dtypes = ('S10', 'f8','f8','f8','f8','f8','f8' )
    frames = frames.group_by('StarName')
    mask = frames.groups.keys['StarName'] == target
    target_data = frames.groups[mask]
    #all_data = frames.groups[mask]
    all_data = Table(names = target_data.colnames, dtype = dtypes)
    keys = frames.groups.keys
    #outfile = os.path.join(cwd, 'Data', 'Outputs', 'lightcurves', 'test.csv')
    targ_x,targ_y = target_data[0]['X'], target_data[0]['Y']


    for i in tqdm(range(len(keys)), desc = 'preprocessing the star: ' + str(target)):

        ensemble_star = keys[i]['StarName']
        key = keys['StarName'] == ensemble_star

        '''if key == target:
            continue'''

        star_data = frames.groups[key]

        #filtering data based on the MJDs in the target data
        new_data = join(target_data, star_data, keys = 'MJD')
        to_remove = [col for col in new_data.colnames if col.endswith('_1')]
        new_data.remove_columns(to_remove)

        #renaming columns to prevent formatting errors
        for col in new_data.colnames:
            new_data.rename_column(col, col.rstrip('_2'))

        #re-ordering columns to prepare for stacking data
        reordered_names = all_data.colnames
        new_data = new_data[reordered_names]

        #checking if the time series for new data is exactly the same as the target data
        #if len(new_data['MJD']) != len(target_data['MJD']):
            #flagged.append(ensemble_star)
            #continue
        if len(new_data['MJD']) == len(target_data['MJD']):
            #combining the data
            all_data = vstack([all_data, new_data])
            x ,y = new_data[0]['X'] ,new_data[0]['Y']
            dist = sqrt((targ_x -x)*(targ_x -x) + (targ_y -y)*(targ_y -y))
            ens_dists.append(dist)
        #print(all_data)
        #write(all_data, outfile, overwrite= True)

    ensemble_data = all_data.group_by('StarName')
    ensemble_keys = list(ensemble_data.groups.keys['StarName'])
    ensemble_keys.remove(target)
    ensemble_data = get_enesemble_data(ensemble_data, ensemble_keys)
    if len(ensemble_data) == 0:
        ensemble_sigs = []
    else:
        ensemble_sigs = get_sigmas(ensemble_data)
    return [all_data, ensemble_sigs]


def fill_col(val, filler):
    #fills empty values in 'val' with 'filler'
    l = [val]
    l.extend(filler)
    return l


def build_table(data):
    '''

    :param data: a list of lists
    :return: an single astropy table of all values
    '''
    time, all_raw_mags, mcor, rng, target_name, num_found, raw_std, cor_std, avg_dist = data
    time = Column(time, name='time', dtype = 'f8')
    all_raw_mags = Column(all_raw_mags, name = 'all_raw_mags', dtype = 'f8')
    mcor = Column(mcor, name = 'mcor', dtype = 'f8')

    Trues = [True for i in range(len(time) - 1)]
    masks = [False]
    masks.extend(Trues)
    int_fill = [0 for i in range(len(Trues))]
    str_fill = ['--' for i in range(len(Trues))]

    #print('something')

    rng = MaskedColumn(fill_col(rng, int_fill), name = 'rng', dtype = 'i4', mask = masks)
    target_name = MaskedColumn(fill_col(target_name,str_fill), name = 'target_name', dtype= 'S50', mask = masks)
    num_found = MaskedColumn(fill_col(num_found, int_fill), name = 'num_found', dtype = 'i4', mask = masks)
    raw_std = MaskedColumn(fill_col(raw_std,int_fill), name = 'raw_std', dtype = 'f8', mask = masks)
    cor_std = MaskedColumn(fill_col(cor_std,int_fill), name = 'cor_std', dtype = 'f8', mask = masks)
    avg_dist = MaskedColumn(fill_col(avg_dist, int_fill), name = 'avg_dist', dtype = 'f8', mask = masks)
    colnames = ['target_name','rng','num_found','raw_std','cor_std','time','all_raw_mags','mcor', 'avg_dist']
    table = Table([target_name,rng,num_found,raw_std,cor_std,time,all_raw_mags,mcor,avg_dist], names = colnames)
    return table


def write_handler(tabled_data, target_name, filename, return_path = False):
    '''

    :param tabled_data: the corrected and raw magnitude with some other data needed for graphing
    :param target_name: sting of the target star
    :param filename: sting name of the file
    :param return_path: Boolean weather to return the sting of the path to the written file
    :return: writes data to a file path but returns the path if return_path == True
    '''


    data_dir = os.path.join(cwd, 'Data', 'Outputs','lightcurves')
    os.makedirs(data_dir, exist_ok= True)
    num = 0
    file_name = ''.join([filename,'_',str(num),'.csv'])
    path = os.path.join(data_dir,file_name)

    while os.path.exists(path):
        num += 1
        file_name = ''.join(['LcGraph_', str(num), '.csv'])
        path = os.path.join(data_dir, file_name)
    write(tabled_data, path, format = 'csv', fast_writer= False)
    if return_path:
        return path


def calc_error_and_name_handler(frames, no_mag_error):
    '''
    :param frames: The data set containing the time series for all observed stars with the Star Name, X and Y corrdinates, magrror,
            and mag
    :param no_mag_error: a boolean Value
    :return: a data table 'frames', with or without the mag error depending on the boolean 'no_mag_error'
    '''
    bad_names = frames.colnames
    if no_mag_error:
        good_names = ['StarName', 'MJD', 'X', 'Y', 'MAG', 'ZeroPoint']

        for bad_name, good_name in zip(bad_names, good_names):
            frames.rename_column(bad_name, good_name)

        mags = frames['MAG']
        mag_err = [(1086/sqrt(1.414*mag))/1000 for mag in mags]
        frames['MagError'] = mag_err

    else:
        good_names = ['StarName', 'MJD', 'X', 'Y', 'MAG','MagError','ZeroPoint']

        for bad_name, good_name in zip(bad_names, good_names):
            frames.rename_column(bad_name, good_name)

    return frames


def filter(frames_files, rng,target, chunk_size = 20, return_data = False,
           return_path = False, write = True, return_avgs_and_sigmas = False,
           no_mag_err = False, plot = True, flagged = None):
    '''
    :param frames_files: A list of files names within the current working directory/Data
    :param rng: integer or float indicating the picture range
    :param target: String name of the target star
    :return: A chart of the filtered and unfiltered data over multiple datasets
    '''

    try:
        from numpy import ndarray as a
    except ImportError:
        print("Library, 'numpy' not installed.")
        sys.exit()

    file_dir = os.path.join(cwd,'Data')
    mjs = []
    merr_ens = []
    allraw_mag_errs = []
    all_raw_mags = []
    star_list = []
    time = []



    '''frames = read(os.path.join(file_dir,frames_files[0]),
                  fast_reader = {'chunk_size': Mb * 1000000}, format = 'csv').group_by('StarName')'''

    #frames = frames_files[0].group_by('StarName')
    #mask = frames.groups.keys['StarName'] == target
    #target_name = frames.groups[mask]['StarName'][0]

    print('Reading File...')
    for frames in frames_files:
        star_keys = None
        print(frames)
        frames = read(os.path.join(file_dir, frames), fast_reader = {'chunk_size':chunk_size * 1000000}, format = 'csv')
        frames = calc_error_and_name_handler(frames, no_mag_err)
        frames = get_kstars2(frames, rng, target, flagged)

        '''if star_keys is None:
            frames = get_kstars2(frames, rng, target)
            star_keys = frames.groups.keys().... efficiency update coming soon!
            should double the speed of the algorithm'''


        frames, ensemble_sigs = preprocess(frames, target, flagged)
        frames = frames.group_by('StarName')

        star_list.extend(frames.groups.keys['StarName'])
        mask = frames.groups.keys['StarName'] == target
        target_data = frames.groups[mask]

        raw_mags = target_data['MAG']
        raw_mag_errs = target_data['MagError']
        frames = frames.group_by('MJD')
        keys = frames.groups.keys
        time.extend(keys['MJD'])

        all_raw_mags.extend(raw_mags)
        allraw_mag_errs.extend(raw_mag_errs)

        for i in range(len(keys)):
            key = keys[i]['MJD']
            key = keys['MJD'] == key
            frame = frames.groups[key]
            mj, merr_j = get_mj(frame)
            mjs.append(mj)
            merr_ens.append(merr_j)


    list_of_lists = [mjs, merr_ens, allraw_mag_errs, all_raw_mags, time]
    list_of_lists = list_to_array(list_of_lists)
    mjs, merr_ens, allraw_mag_errs, all_raw_mags, time = flatten_arrays(list_of_lists)


    corerr = [sqrt(merr_i * merr_i + merens * merens) for merr_i, merens in zip(allraw_mag_errs, merr_ens)]

    num_found = len(set(star_list))
    num_frames = len(time)
    if num_frames ==0:
        print('Error: No stars were found for the ensemble, this star might not appear in all plates.')
        return None
    M = sum(mjs) / num_frames

    mcor = [mag - (mj - M) for mag, mj in zip(all_raw_mags, mjs)]
    avg_cor_mag = np.mean(mcor)
    avg_raw_mag = np.mean(all_raw_mags)
    raw_std, cor_std = list(map(np.std, [all_raw_mags, mcor]))
    num_frames = len(time)
    target_name = target

    print("number of stars found", num_found)

    '''return {'raw_sigma': raw_std, 'corrected_sigma': cor_std, 'num_found': num_found, 'time': time}'''

    data = [time, all_raw_mags, mcor, rng, target_name, num_found, raw_std, cor_std]
    tabled_data = build_table(data)

    path = None
    if write:
        path = write_handler(tabled_data, target_name, 'LcUnplotted', return_path=True)
    if return_data:
        return [time, all_raw_mags, mcor, rng, target_name, num_found, raw_std, cor_std]
    if return_path:
        if path is None:
            print('Warning: path returns only when param: write = True')
        return path
    if return_avgs_and_sigmas:
        return avg_cor_mag, avg_raw_mag, raw_std, cor_std
    if plot:
        if path is None:
            print("Can't plot because there is no path, make sure write = True")
            sys.exit()
        plot_data4(path)


def plot_data4(file_path):

    '''

    :param file_path: path of the file for the data
    :return: returns None, plots the data from the written file path
    '''
    list_of_datas = read(file_path,
                  fast_reader = {'chunk_size': Mb * 1000000}, format = 'csv')

    target_name, rng, raw_std, cor_std, num_found, avg_dist = list_of_datas[0]['target_name', 'rng', 'raw_std', 'cor_std', 'num_found', 'avg_dist']
    rng = rng*6.3/60
    time = list_of_datas['time']
    all_raw_mags = list_of_datas['all_raw_mags']
    mcor = list_of_datas['mcor']
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    raw_scat = ax.scatter(time, all_raw_mags, alpha=0.5, label='Raw Magnitudes', marker='x', c='blue')
    cor_scat = ax.scatter(time, mcor, alpha=0.5, label='Corrected Magnitudes', marker='x', c='red')
    text = 'Raw Sigma: {:1.4f} \nCorrected Sigma: {:1.4f} \nStars Averaged: {:1.0f}\n Range: {:1.0f}\n Avg Distance: {:1.3f}'.format(raw_std, cor_std, num_found, rng, avg_dist)
    ax.legend([extra, raw_scat, cor_scat], (text, "Raw Magnitudes", "Corrected Magnitudes"),bbox_to_anchor=(0.00, 1), loc=2, borderaxespad=0.)
    #ax.set_ylim(ymin = 9.5, ymax = 9.8)
    #ax.set_title(''.join(['Light Curve of ', str(target_name), 'in Range: ', str(list_of_datas['rng']), 'Pixels']))
    ax.set_title(str(target_name))
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude')
    # ax.legend([raw_scat, cor_std], ['Raw', 'Corrected'])
    data_dir = os.path.join(cwd, 'Data', 'Outputs', 'graphs')
    filename = str(target_name)+'_plotted.png'
    path = os.path.join(data_dir,filename)
    plt.savefig(path)
    plt.show()



def merge(list_of_files, chunk_size = 20, dtypes = None, merged = None):
    '''

    :param list_of_files: a list of file names
    :param chunk_size: chunck size for the file reader
    :param dtypes: the set of datatypes for each column of the new dataset
    :param merged: A table to add the data too
    :return: none, writes merged set to outfile
    '''
    if dtypes is None:
        dtypes = ('S10', 'f8','f8','f8','f8','f8','f8')
    for file in tqdm(list_of_files):
        path = os.path.join(cwd,'Data', file)
        frames = read(path, fast_reader = {'chunk_size':chunk_size * 1000000}, format = 'csv')
        if merged is None:
            merged = Table(names = frames.colnames, dtype = dtypes)
        merged = vstack([merged,frames])
    outfile = os.path.join(cwd, 'Data', 'E_W_merged.csv')
    write(merged, outfile, format = 'csv')


def get_all_mjs(frames):
    '''

    :param frames: the data with the observed stars
    :return: a list of the mjs and there corresponding error
    '''

    mjs = []
    merr_ens = []
    frames = frames.group_by('MJD')
    keys = frames.groups.keys
    for i in range(len(keys)):
        key = keys[i]['MJD']
        key = keys['MJD'] == key
        frame = frames.groups[key]
        mj, merr_j = get_mj(frame)
        mjs.append(mj)
        merr_ens.append(merr_j)
    return mjs, merr_ens


def get_vals(frames, target):
    '''

    :param frames: the data with the observed stars
    :param target: the name of the target star
    :return: the raw magnituds, errors, and MJDs as lists
    '''
    frames = frames.group_by('StarName')
    mask = frames.groups.keys['StarName'] == target
    target_data = frames.groups[mask]
    all_mag_errs = target_data['MagError']
    all_raw_mags = target_data['MAG']
    frames = frames.group_by('MJD')
    times = frames.groups.keys
    time = times['MJD']

    return all_raw_mags, all_mag_errs, time



def remove_high_sigmas(ensemble_sigs, M):
    '''

    :param ensemble_sigs: a list of sigmas for each ensemble star
    :param M: the weighted avererage sigma (see astrokit paper)
    :return: a list of new ensemble sigmas and stars that have been flagged for having a sigma grater than 2*M
    '''
    more_flagged = []
    for key in ensemble_sigs:
        if ensemble_sigs[key] > 2*M:
            more_flagged.append(key)
            del ensemble_sigs[key]
    return ensemble_sigs, more_flagged



def auto_filter(frames, target, chunk_size = 20, return_data = False, return_path = False, write = True,
                return_avgs_and_sigmas = False, no_mag_err = False, plot = True, max_ensemble_size = 10, max_rng = 30, start_rng = 5.0, increase = 1):
    '''
        :param frames_files: A list of files names within the current working directory/Data
        :param rng: integer or float indicating the picture range
        :param target: String name of the target star
        :return: A chart of the filtered and unfiltered data over multiple datasets
        '''
    ensemble_size = 0

    max_rng = max_rng * 60/6.3 #converting arc miniutes to pixels

    rng = start_rng * 60 / 6.3 #converting arc miniutes to pixels
    pincrease = increase * 60 / 6.3
    flagged = []
    mjs = []
    time = []
    all_raw_mag_errs = []
    all_raw_mags = []
    ens_dists = []
    M = 0
    in_max_range = None

    og_frames = frames
    while ensemble_size < max_ensemble_size and rng < max_rng:
        if in_max_range is None:
            in_max_range = get_kstars2(target= target, frames = frames,rng=max_rng,flagged=flagged,max_ensemble_size= max_ensemble_size)
            frames = in_max_range
        #frames = og_frames
        frames = get_kstars2(frames, rng, target, flagged, max_ensemble_size)
        frames = frames.group_by('StarName')
        frames, ensemble_sigs = preprocess(frames, target, flagged, ens_dists)

        if len(ensemble_sigs) == 0:
            print('None found, increasing range by ',increase,' arc miniute(s).')
            rng += pincrease
            continue


        #mask = frames.groups.keys['StarName'] == target
        #target_data = frames.groups[mask]

        all_raw_mags, all_raw_mag_errs, time = get_vals(frames, target)
        mjs, merr_ens = get_all_mjs(frames)

        list_of_lists = [mjs, merr_ens, all_raw_mag_errs, all_raw_mags, time]
        list_of_lists = list_to_array(list_of_lists)
        mjs, merr_ens, all_raw_mag_errs, all_raw_mags, time = flatten_arrays(list_of_lists)
        num_frames = len(time)
        M = sum(mjs) / num_frames

        ensemble_sigs, more_flagged = remove_high_sigmas(ensemble_sigs, M)


        flagged.extend(more_flagged)

        if len(more_flagged) != 0:
            print('Found more flagged')
            print('New Flagged List:', flagged)


        ensemble_size = len(ensemble_sigs)

        if ensemble_size < max_ensemble_size:
            print('Ensemble too small, increasing range by ',increase ,' arc miniute(s).')
            rng += pincrease


    num_found = ensemble_size
    num_frames = len(time)
    if num_frames ==0:
        print('Error: No stars were found for the ensemble, this star may not appear in all plates.')
        return None
   # M = sum(mjs) / num_frames
    mcor = [mag - (mj - M) for mag, mj in zip(all_raw_mags, mjs)]
    avg_cor_mag = np.mean(mcor)
    avg_raw_mag = np.mean(all_raw_mags)
    raw_std, cor_std = list(map(np.std, [all_raw_mags, mcor]))
    target_name = target
    avg_dist = np.mean(ens_dists) * 6.3/ 60
    data = [time, all_raw_mags, mcor, rng, target_name, num_found, raw_std, cor_std,avg_dist]
    tabled_data = build_table(data)

    path = None
    if write:
        path = write_handler(tabled_data, target_name, 'LcUnplotted', return_path=True)
    if return_data:
        return [time, all_raw_mags, mcor, rng, target_name, num_found, raw_std, cor_std, ens_dists]
    if return_path:
        if path is None:
            print('Warning: path returns only when param: write = True')
        return path
    if return_avgs_and_sigmas:
        return avg_cor_mag, avg_raw_mag, raw_std, cor_std, ens_dists
    if plot:
        if path is None:
            print("Can't plot because there is no path, make sure write = True")
            sys.exit()
        plot_data4(path)





    def plot(mags, sigs, dists):
        fig = plt.figure()
        sub1 = fig.add_subplot(111)
        sub1.plot(mags,sigs, 'rx', label = "Standard Deviation")
        sub1.set_ylabel('Average Sigma')
        sub1.set_xlabel('Average Corrected Magnitude')
        sub1.set_title('Standard Deviation vs Magnitude')
        sub2 = fig.add_subplot(211)
        sub2.set_ylabel('Average Sigma')
        sub2.set_xlabel('Average Ensmeble Distance')
        sub2.set_title('Standard Deviation vs Average Ensemble Distance')
        sub2.plot(dists,sigs)
        plt.savefig(os.path.join(cwd, 'Data', 'sigma_vs_mag_vs_dist.png'))
        plt.show()

    if plot:
        plot(avg_cor_mags,cor_sigmas, avg_dists)




def auto_filter_multiple(frames, list_of_targets, chunk_size = 20, return_data = False, return_path = False, write = True,
                return_avgs_and_sigmas = False, no_mag_err = False, plot = True, max_ensemble_size = 10, max_rng = 30, start_rng = 5.0, increase = 1):

    for target in list_of_targets:
        auto_filter(frames,target, chunk_size, return_data, return_path, write, max_ensemble_size = max_ensemble_size, max_rng = max_rng, start_rng= start_rng, increase = increase)
