"""
"""

import sys
import math
import pathlib
import napari
import numpy as np
import torch as pt
from magicgui import magicgui
from pathlib import Path
from tqdm import tqdm
from napari_cool_tools_io import device
from napari_cool_tools_io._prof_reader import prof_proc_meta


def dc_subtraction_double_sweep_pt(data):
    """this function is used to remove the dc signal (dc substraction) for double sweep source signal
    therefore, the number of a-scan per b-scan is expected to be even number otherwise, this function will produce error message"""

    # prep temorary array

    #print(type(data.shape[0]))

    corrected_1 = pt.zeros((int(data.shape[0]/2),data.shape[1]))
    corrected_2 = pt.zeros((int(data.shape[0]/2),data.shape[1]))

    #print(f"corrected 1 shape: {corrected_1.shape}\ncorrected 2 shape: {corrected_2.shape}\n")

    # flip odd data
    
    #print(data,data.shape)
    #print(data[1::2,:],data[1::2,:].shape)
    #print(data[1::2,:][::-1,:],data[1::2,:][::-1,:].shape)

    corrected_1 = data[::2,:]#[:,::-1]
    corrected_1 = corrected_1.flip(1)
    corrected_2 = data[1::2,:]  

    # subtract for each spectrum
    corrected_1 = corrected_1 - pt.tile(corrected_1.median(0)[0],(corrected_1.shape[0],1)) #pt.tile(pt.median(corrected_1,0),(corrected_1.shape[0],1))
    corrected_2 = corrected_2 - pt.tile(corrected_2.median(0)[0],(corrected_2.shape[0],1)) #pt.tile(pt.median(corrected_2,0),(corrected_2.shape[0],1))

    # recombine even and odd rows into b-scan
    subtracted_signal = pt.zeros(data.shape)
    subtracted_signal[::2,:] = corrected_1
    subtracted_signal[1::2,:] = corrected_2

    #print(f"subtracted signal: {subtracted_signal}")

    return subtracted_signal

def set_displacement_coefficients(data, maxDispOrders, coefRange):
    """"""
    arrCountDispCoeff = np.zeros((maxDispOrders-1,1))

    for idx_CounterDispCoef in tqdm(range(0,len(arrCountDispCoeff)),desc="Calculating Displacement Coefficients"):
        arrDispCoeffRange = np.arange(-1*coefRange,coefRange+1 ,1)
        arrCost = np.zeros((arrDispCoeffRange.shape[0]))

        for k in tqdm(range(0,len(arrDispCoeffRange)),desc="Calculating Costs"):
            arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[k]

            #print(f"arrCountDispCoeff: {arrCountDispCoeff}\n")

            arrCost[k] = cal_cost_function(data,maxDispOrders,arrCountDispCoeff)

        argMinCost = arrCost.argmin()
        arrCountDispCoeff[idx_CounterDispCoef] = arrDispCoeffRange[argMinCost] 

        #print(f"arrCost shape: {arrCost.shape}\n")
        #print(f"argMinCost: {argMinCost}\narrCostMin: {arrCost.min()}\n")

    return arrCountDispCoeff

def cal_cost_function(data, maxDispOrders, arrCountDispCoeff):
    """"""
    data_disp_comp = comp_dis_phase(data, maxDispOrders,arrCountDispCoeff)

    oct = np.abs(np.fft.fft(data_disp_comp))**2
    roi_oct = oct[:,49:int(data.shape[1]/2)-50] # avoid edges
    norm_oct = roi_oct/np.sum(roi_oct[:])
    entropy = -1*(norm_oct*np.log10(norm_oct)) # measure th entropy to get the shaprness
    cost = np.sum(entropy[:]) #%shenon entropy

    return cost

def comp_dis_phase(data,maxDsipOrders,arrCountDispCoeff):
    """"""
    scan_pts = data.shape[1]
    line_per_frame = data.shape[0]
    k_linear = np.linspace(-1,1,scan_pts)
    k_axis = np.tile(k_linear, (line_per_frame,1)) -1

    #print(f"scan pts: {scan_pts}\n")
    #print(f"line per frame: {line_per_frame}\n")
    #print(f"k_linear {k_linear}\n")
    #print(f"kaxis shape: {k_axis.shape}\nkaxis: {k_axis}\n")

    amp = np.abs(data)
    phase = np.angle(data)

    #print(f"amp: {amp}\nphase: {phase}\n")

    for i in range(maxDsipOrders-1):
        phase = phase + arrCountDispCoeff[i]*(k_axis**(i+2))

        #print(f"arrCountDispCoeff[i]: {arrCountDispCoeff[i]}")
        #print(f"k_axis**(i+1): {k_axis**(i+2)}\n")
        #print(f"new pahse: {phase}\n")    

    data_disp_comp = amp*np.exp(1j*phase)

    #print(f"data_disp_comp: {data_disp_comp}\n")

    return data_disp_comp

        

@magicgui(
    unp_file_path={"label": ".unp file", "mode": "r"},
    call_button="Process .unp file",
)
def process_unp(
    unp_file_path: pathlib.Path = Path("D:\\John\\Yakub\\Shuibin\\Dispersion Correction\\SimpleDispersionCorrection\\14_09_04.unp"),
    pytorch:bool=True,
    GPU:bool=True
    ):
    """
    """
    
    file_name = unp_file_path.name
    folder = unp_file_path.parent
    file_type = unp_file_path.suffix

    if file_type == ".unp":
        print(f"{file_name}\n{folder}\n{unp_file_path}\n")

        # Read the xml file
        meta = prof_proc_meta(Path(unp_file_path),".unp")

        print(meta)

        if meta is not None:
            h, w, d, bmscan, w_param, dtype, layer_type = meta

            if bmscan > 1:
                bidir = "false"
                bidir_a = "false"
            else:
                bidir = "true"
                bidir_a = "true"

            meta_params = {
                "width": w_param,
                "height": h,
                "frames": d,
                "bidir": bidir,
                "bidir_a": bidir_a,
                "bscan_width":w,
                "bmscan": bmscan
            }

            print(f"meta params {meta_params}\n")

        '''
        reference_frame = math.ceil(meta_params["frames"]/2)
        center_frame_idx = 2*meta_params["width"]*meta_params["height"]*(reference_frame-1)
        items_to_read = meta_params["width"]*meta_params["height"]

        center_frame = np.fromfile(unp_file_path,dtype=np.uint16,count=items_to_read,offset=center_frame_idx)
        
        center_frame = center_frame.reshape((-1,meta_params["height"],meta_params["width"]))

        print(f"center frame size: {center_frame.size}\ncenter frame itemsize: {center_frame.itemsize}\ncenter_frame shape: {center_frame.shape}\ncenter_frame dtype: {center_frame.dtype}\n")
        print(f"center frame memory size: {center_frame.size*center_frame.itemsize}")
        '''

        oct_vol_array = []

        # open file
        with open(unp_file_path,'rb',buffering=0) as byte_reader:
            
            # Set reference A-scan to0 find the dispersion coefficients
            # Use center frame (b-scan) of the volume
            reference_frame = math.ceil(meta_params["frames"]/2)
            print(f"referemce frame: {reference_frame}\n")

            # get idx of center frame (bscan)
            center_frame_idx = 2*meta_params["width"]*meta_params["height"]*(reference_frame-1)
            print(f"center frame index: {center_frame_idx}\n")
            # move to center frame in binary file
            byte_reader.seek(center_frame_idx,0)
            # read 2 bytes at a time (uint16)

            data_size_bytes = 2*meta_params["width"]*meta_params["height"]
            ref_RawData = byte_reader.read(data_size_bytes)
            array = np.frombuffer(ref_RawData,dtype=np.uint16)
            array = array.reshape((-1,meta_params["height"],meta_params["width"]))
            new_array = array.astype(np.int64)
            torch_array = pt.from_numpy(new_array)

            print(f"pytorch array: {torch_array}\n")

            '''
            print(f"numpy array size: {array.size}\nnumpy itemsize: {array.itemsize}\nnumpy shape: {array.shape}\nnumpy dtype: {array.dtype}\n")
            print(f"new numpy array size: {new_array.size}\nnew numpy itemsize: {new_array.itemsize}\nnew numpy shape: {new_array.shape}\nnew numpy dtype: {new_array.dtype}\n")
            print(f"array memory size: {array.size*array.itemsize}")
            print(f"new array memory size: {new_array.size*new_array.itemsize}")
            
            print(f"python system size: {sys.getsizeof(ref_RawData)}, type: {type(ref_RawData)}\n")
            print(f"new python system size: {sys.getsizeof(new_array)}, type: {type(new_array)}\n")
            print(f"Array values: {array[0]}\n")
            print(f"New array values: {new_array[0]}\n")
            '''

            # Subtract the DC signal
            #Here we use special function to remove dc signal for each odd and even a-scan spectrum %%%
            #This is because we are using the double side sweeping laser source %%%

            subtracted_signal = dc_subtraction_double_sweep_pt(torch_array.squeeze())

            print(f"subtracted_signal_pt: {subtracted_signal}\n")
            
            
            # Hamming windowing

            #temp = np.tile(np.hamming(subtracted_signal.shape[1]),(subtracted_signal.shape[0],1))
            hamming = pt.tile(pt.hamming_window(subtracted_signal.shape[1]),(subtracted_signal.shape[0],1)) #hamming = np.tile(np.hamming(subtracted_signal.shape[1]),(subtracted_signal.shape[0],1))
            
            exit()
            #temp = (subtracted_signal * hamming)[:,::-1]
            #print(temp)

            hamming_signal = (subtracted_signal * hamming)[:,::-1]

            # Brute force dispersion coefficient finder
            # This method is used because it can be forwardly transofrmed into parallel computing in GPU %%%

            dispMaxOrder = 3
            coeffRange = 100
            #dispCoeffs = set_displacement_coefficients(hamming_signal,dispMaxOrder,coeffRange)
            dispCoeffs = np.array([29,0]) # PUT THIS BACK AFTER TESTING!!!!

            # Main OCT Volume process
            
            for frame_num in tqdm(range(0,meta_params["frames"]),desc="Processing Bscans"):
            #for frame_num in range(1199,1200):
            #for frame_num in range(1199,1202):

                current_idx = 2*meta_params["width"]*meta_params["height"]*(frame_num)
                byte_reader.seek(current_idx,0)
                raw_data = np.frombuffer(byte_reader.read(data_size_bytes),dtype=np.uint16)
                temp_raw = raw_data.reshape((-1,meta_params["height"],meta_params["width"]))
                raw = temp_raw.astype(np.uint64)

                # Subtract the DC signal
                subtracted_signal = dc_subtraction_double_sweep(raw.squeeze())

                # Hamming windowing
                hamming_signal = (subtracted_signal * hamming)[:,::-1]

                # Dispersion Correction
                img_disp_comp = comp_dis_phase(hamming_signal,dispMaxOrder,dispCoeffs)

                # Fourier Transform
                column_pad = np.array([0,int((2**11 - img_disp_comp.shape[1])/2)])
                img_disp_comp_pad = np.pad(np.squeeze(img_disp_comp), ((column_pad[0],column_pad[0]),(column_pad[1],column_pad[1])))
                fft_signal = np.fft.fft(img_disp_comp_pad)

                # double sided fast axis scans
                temp_frame = np.abs(fft_signal[:,:int(fft_signal.shape[1]/2)])
                if (np.floor(frame_num/meta_params["bmscan"])%2):
                    temp_frame = temp_frame[::-1,:]
                    #print("mod flip temp_frame\n")
                else:
                    pass

                # (optional) flip the image
                temp_frame = temp_frame[:,::-1]
                
                # store image to oct array
                oct_vol_array.append(temp_frame)
                #print("Here")

            volOCT = np.stack(oct_vol_array,axis=0)
            #print(f"OCT volume: {volOCT[1199,:,:]}\n")

            m_scan_reshape = volOCT.reshape((-1,meta_params["bmscan"],volOCT.shape[-2],volOCT.shape[-1]))
            print(f"m_scan_reshape shape: {m_scan_reshape.shape}\n")
            volOCT_avg = m_scan_reshape.mean(1)
            print(f"volOCT_avg shape: {volOCT_avg.shape}\n")
            #print(f"volOCT_avg: {volOCT_avg[399,:,:]}\n")
            print(f"volOCT_avg: {volOCT_avg[1,:,:]}\n")

            #for i,scan in enumerate(volOCT):
            #for i,scan in enumerate(volOCT_avg):
                #temp = volOCT[1199,:,:]
                #temp = volOCT[i,:,:]
            temp = volOCT_avg[:,:,:]

            # convert to logarithmic scale
            temp = 20*np.log10(np.abs(temp))

            # remove infinite vlaues
            temp[np.isinf(temp)] = 0

            # normalize the image
            temp_max_val = temp.max()
            temp_min_val = temp.min()
            #volOCT[i,:,:] = (temp-temp_min_val)/(temp_max_val-temp_min_val)
            volOCT_avg[:,:,:] = (temp-temp_min_val)/(temp_max_val-temp_min_val)

            #print(f"volOCT_avg_log: {volOCT_avg[399,:,:]}\n")
            print(f"volOCT_avg_log: {volOCT_avg[1,:,:]}\n")

            # switch from columm major to row major
            #volOCT_log_transp = volOCT.transpose(0,2,1)
            volOCT_avg_log_transp = volOCT_avg.transpose(0,2,1)

            #print(f"volOCT_avg_log_transpose: {volOCT_avg_log_transp[399,:,:]}\n")

            viewer = napari.Viewer()
            #viewer.add_image(volOCT_log_transp)
            viewer.add_image(volOCT_avg_log_transp)

        print(f"{unp_file_path} file processing is finished.")
    else:
        print(f"File must be of type '.unp'. {file_name} is not the proper type.")


process_unp.show(run=True)