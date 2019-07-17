from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from bathyml.keras.layers import *
from typing import List, Optional, Tuple, Dict, Any, Union
import time, copy, os
from bathyml.keras.processing import Analytics, Parser
import tensorflow as tf
from datetime import datetime
from keras.callbacks import TensorBoard, History, Callback
import matplotlib.pyplot as plt
import multiprocessing as mp
import random, sys
import numpy as np
from keras.optimizers import SGD
import logging, traceback
outDir = os.path.join( os.path.expanduser("~"), "results" )
LOG_FILE = os.path.join( outDir, 'logs', 'cliMLe.log' )
try: os.makedirs( os.path.dirname(LOG_FILE) )
except: pass
RESULTS_DIR = os.path.join( outDir, 'results' )
try: os.makedirs( RESULTS_DIR )
except: pass
logging.basicConfig(filename=LOG_FILE,level=logging.DEBUG)

class FitResult(object):

    @staticmethod
    def getCombinedAverages( results ):
        # type: (list[FitResult]) -> ( np.ndarray, np.ndarray, int )
        ave_train_loss_sum = None
        ave_val_loss_sum = None
        nInstances = 0
        for result in results:
            ( ave_loss_history, ave_val_loss_history, nI ) = result.getAverages()
            ave_train_loss_sum = Analytics.intersect_add( ave_train_loss_sum, ave_loss_history * nI )
            ave_val_loss_sum =  Analytics.intersect_add(  ave_val_loss_sum,   ave_val_loss_history * nI )
            nInstances += nI
        return (None,None,nInstances) if ave_train_loss_sum is None else ( ave_train_loss_sum / nInstances, ave_val_loss_sum / nInstances, nInstances )

    @staticmethod
    def new( history: History, initial_weights: List[np.ndarray], final_weights: List[np.ndarray], training_loss: float, val_loss: float,  nEpocs: int ) -> "FitResult":
        return FitResult( history.history['val_loss'], history.history['loss'], initial_weights, final_weights,  training_loss, val_loss, nEpocs )

    def __init__( self, _val_loss_history, _train_loss_history, _initial_weights, _final_weights, _training_loss,  _val_loss, _nEpocs ):
        self.val_loss_history = np.array( _val_loss_history )
        self.train_loss_history = np.array( _train_loss_history )
        self.initial_weights = _initial_weights
        self.final_weights = _final_weights
        self.val_loss = float( _val_loss )
        self.train_loss = float( _training_loss )
        self.nEpocs = int( _nEpocs )
        self.ave_train_loss_history = None
        self.ave_val_loss_history = None
        self.nInstances = -1

    def getAverages(self):
        return (self.ave_train_loss_history, self.ave_val_loss_history, self.nInstances)

    def recordPerformance( self, performanceTracker: "PerformanceTracker" )-> "FitResult":
        (self.ave_train_loss_history, self.ave_val_loss_history) = performanceTracker.getHistory()
        self.nInstances = performanceTracker.nIters
        return self

    def setPerformance( self, results ):
        # type: (list[FitResult]) -> FitResult
        (self.ave_train_loss_history, self.ave_val_loss_history, self.nInstances) = FitResult.getCombinedAverages(results)
        print(( "Setting performance averages from {0} instances".format(self.nInstances) ))
        return self

    def getScore( self, score_val_weight ):
        return score_val_weight * self.val_loss + self.train_loss

    def serialize( self, lines ):
        lines.append( "#Result" )
        Parser.sparm( lines, "val_loss", self.val_loss )
        Parser.sparm( lines, "train_loss", self.train_loss )
        Parser.sparm( lines, "nepocs", self.nEpocs )
        Parser.sparm(lines, "ninstances", self.nInstances)
        Parser.sarray( lines, "val_loss_history", self.val_loss_history )
        Parser.sarray( lines, "train_loss_history", self.train_loss_history )
        Parser.sarray(lines, "ave_train_loss_history", self.ave_train_loss_history)
        Parser.sarray( lines, "ave_val_loss_history", self.ave_val_loss_history )
        Parser.swts( lines, "initial", self.initial_weights )
        Parser.swts( lines, "final", self.final_weights )

    @staticmethod
    def deserialize( lines ):
        active = False
        parms = {}
        arrays = {}
        weights = {}
        for line in lines:
            if line.startswith("#Result"):
                active = True
            elif active:
                if line.startswith("#"): break
                elif line.startswith("@P:"):
                    toks = line.split(":")[1].split("=")
                    parms[toks[0]] = toks[1].strip()
                elif line.startswith("@A:"):
                    toks = line.split(":")[1].split("=")
                    arrays[toks[0]] = [ float(x) for x in toks[1].split(",") ]
                elif line.startswith("@W:"):
                    toks = line.split(":")[1].split("=")
                    wts = []
                    for wtLine in toks[1].split(";"):
                        wtToks = wtLine.split("|")
                        shape = [int(x) for x in wtToks[0].split(",")]
                        data = [float(x) for x in wtToks[1].split(",")]
                        wts.append( np.array(data).reshape(shape) )
                    weights[toks[0]] = wts
        rv = FitResult( arrays["val_loss_history"], arrays["train_loss_history"], weights["initial"], weights["final"], parms["train_loss"],  parms["val_loss"], parms["nepocs"] )
        rv.ave_val_loss_history = arrays.get("ave_val_loss_history",None)
        if rv.ave_val_loss_history is not None:
            rv.ave_val_loss_history = np.array( rv.ave_val_loss_history )
        rv.ave_train_loss_history = arrays.get("ave_train_loss_history", None)
        if rv.ave_train_loss_history is not None:
            rv.ave_train_loss_history = np.array( rv.ave_val_loss_history )
        rv.nInstances = parms.get("ninstances",-1)
        return rv


    def valLossHistory(self):
        return self.val_loss_history

    def lossHistory(self):
        return self.train_loss_history

    def isMature(self):
        return self.val_loss < self.train_loss

    def __lt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss < other.val_loss

    def __gt__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss > other.val_loss

    def __eq__(self, other ):
        # type: (FitResult) -> bool
        return self.val_loss == other.val_loss

    @staticmethod
    def getBest( results ):
        # type: (list[FitResult]) -> FitResult
        bestResult = None  # type: FitResult
        for result in results:
            if isinstance(result, str):
                raise Exception( "A worker raised an Exception: " + result )
            elif result and result.isMature:
                if bestResult is None or result < bestResult:
                    bestResult = result
        return bestResult.setPerformance( results )


class PerformanceTracker(Callback):
    def __init__( self, _stopCond, **kwargs ):
        super(PerformanceTracker,self).__init__()
        self.stopCond = _stopCond
        self.val_loss_history = None
        self.training_loss_history = None
        self.nIters = 0
        self.verbose = kwargs.get( "verbose", False )
        self.termIters = kwargs.get('earlyTermIndex', -1 )

    def on_train_begin(self, logs=None):
        self.minValLoss = sys.float_info.max
        self.minTrainLoss = sys.float_info.max
        self.nSteps = 0
        self.nEpoc = 0
        self.epochCount = 0
        self.best_weights = None
        self.nUphillIters = -1

    def on_train_end(self, logs=None):
        self.nIters += 1
        val_loss = np.array(self.model.history.history['val_loss'])
        training_loss = np.array(self.model.history.history['loss'])
        self.val_loss_history = Analytics.intersect_add( self.val_loss_history, val_loss )
        self.training_loss_history = Analytics.intersect_add(self.training_loss_history, training_loss)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = self.model.history.history.get('val_loss',None)
        training_loss = self.model.history.history.get('loss',None)
        if self.verbose:
            tloss = training_loss[-1] if training_loss else "UNDEF"
            vloss = val_loss[-1] if val_loss else "UNDEF"
            print("* I[{0}]-E[{1}]-> training_loss: {2}, val_loss: {3}".format( self.nIters, self.epochCount, tloss, vloss ))
        if val_loss and training_loss:
            vloss, tloss = val_loss[-1], training_loss[-1]
            self._processIter( vloss, tloss )
        if ( self.termIters > 0 ) and ( self.nUphillIters >= self.termIters ):
            logging.info( "Stopping training at iteration: " + str( self.nIters ) )
            self.model.stop_training = True
        self.epochCount = self.epochCount + 1

    def _processIter(self, valLoss, trainLoss ):
        self.nSteps += 1
        print( f"processIter[{self.nSteps}], valLoss = {valLoss}, trainLoss = {trainLoss}, minValLoss = {self.minValLoss}")
        if (valLoss < self.minValLoss) :
            if (self.stopCond == "minVal") or ((self.stopCond == "minValTrain") and (trainLoss <= valLoss)):
                self.minValLoss = valLoss
                self.minTrainLoss = trainLoss
                self.nEpoc = self.nSteps
                self.nUphillIters = 0
                weights = self.model.get_weights()
                self.best_weights = copy.deepcopy(weights)
                print(f"  ---> update Weights[{self.nEpoc}], weights[0][0][0:5] = {weights[0][0][0:5]}")
        elif self.nUphillIters >= 0:
            self.nUphillIters += 1

    def getHistory(self):
        return (self.training_loss_history / self.nIters, self.val_loss_history / self.nIters)

    def getWeights(self):
        return self.best_weights

class ActivationTarget:
    def __init__(self, target_value, title_str ):
        self.value = target_value
        self.title = title_str

class LearningModel(object):

    def __init__( self, inputDataset: np.ndarray, trainingDataset: np.ndarray, _layers=None, **kwargs ):
        self.inputData = inputDataset
        self.outputData = trainingDataset
        self.parms = kwargs
        self.batchSize = int( kwargs.get( 'batch', 50 ) )
        self.layers = _layers                               # type: list[Layer]
        self.nEpocs = int( kwargs.get( 'epocs', 300 ) )
        self.validation_fraction = float( kwargs.get(  'vf', 0.1 ) )
        self.timeRange = kwargs.get(  'timeRange', None )
        self.eager = kwargs.get(  'eager', False )
        self.hidden = Parser.raint( kwargs.get( 'hidden', None ) )
        self.activation = kwargs.get( 'activation', 'relu' )
        self.shuffle = bool( kwargs.get('shuffle', False ) )
        self.lossFunction = kwargs.get('loss_function', 'mse' )
        self.weights = Parser.ro( kwargs.get( 'weights', None ) )
        self.lr = float( kwargs.get( 'lrate', 0.01 ) )
        self.stop_condition = kwargs.get('stop_condition', "minValTrain")
        self.momentum = float( kwargs.get( 'momentum', 0.0 ) )
        self.decay = float( kwargs.get( 'decay', 0.0 ) )
        self.nesterov = bool( kwargs.get( 'nesterov', False ) )
        self.timestamp = datetime.now().strftime("%m-%d-%y.%H:%M:%S")
        self.logDir = os.path.expanduser("~/results/logs/{}".format( "bathyml_" + self.timestamp ) )
        self.tensorboard = TensorBoard( log_dir=self.logDir, histogram_freq=0, write_graph=True )
        self.performanceTracker = PerformanceTracker( self.stop_condition, **kwargs )
        self.bestFitResult = None # type: FitResult
        self.layer_instances = []

    def layerSize(self, layer_index=0 ):
        return self.hidden[layer_index] if self.hidden is not None else self.layers[layer_index].dim

    def execute( self, nIterations=1, shuffle=True, procIndex=-1 )-> FitResult:
        self.reseed()
        for iterIndex in range(nIterations):
            model = self.createSequentialModel()  # type: Sequential
            initial_weights = model.get_weights()
#            history: History = model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, validation_split=self.validation_fraction, shuffle=shuffle, callbacks=[self.tensorboard,self.performanceTracker], verbose=0 )
            history: History = model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=self.nEpocs, callbacks=[ self.tensorboard,self.performanceTracker ] )
            self.updateHistory( history, initial_weights, iterIndex, procIndex )
        return self.bestFitResult.recordPerformance( self.performanceTracker )

    def shuffle_data(self, input_data, training_data ): #  -> ( shuffled_input_data, shuffled_training_data):
        indices = np.arange(input_data.shape[0])
        np.random.shuffle(indices)
        shuffled_input_data, shuffled_training_data = np.copy(input_data), np.copy(training_data)
        shuffled_input_data[:] = input_data[indices]
        shuffled_training_data[:] = training_data[indices]
        return ( shuffled_input_data, shuffled_training_data)

    @staticmethod
    def deserialize( inputDataset, trainingDataset, lines ):
        active = False
        parms = {}
        layers = None
        for line in lines:
            if line.startswith("#LearningModel"):
                active = True
            elif active:
                if line.startswith("#"): break
                elif line.startswith("@P:"):
                    toks = line.split("P:")[1].split("=")
                    if toks[0] == "layers":
                        layers = Layers.deserialize( toks[1] )
                    else:
                        parms[toks[0]] = toks[1].strip()
        return LearningModel( inputDataset, trainingDataset, layers, **parms )

    # @classmethod
    # def load( cls, path ):
    #     # type: (str) -> ( LearningModel, FitResult )
    #     file = open( path, "r")
    #     lines = file.readlines()
    #     inputDataset = InputDataset.deserialize(lines)
    #     trainingDataset = TrainingDataset.deserialize( lines )
    #     learningModel = LearningModel.deserialize( inputDataset, trainingDataset, lines )
    #     result = FitResult.deserialize( lines )
    #     return ( learningModel, result )
    #
    # @classmethod
    # def loadInstance( cls, instance ):
    #     # type: (str) -> ( LearningModel, FitResult )
    #     return cls.load( os.path.join( RESULTS_DIR, instance ) )

    def reseed(self):
        seed = random.randint(0, 2 ** 32 - 2)
        np.random.seed(seed)


    # def getRandomWeights(self):
    #     self.reseed()
    #     new_weights = []
    #     for wIndex in range(len(self.weights_template)):
    #         wshape = self.weights_template[wIndex].shape
    #         if wIndex % 2 == 1:     new_weights.append(  2 * np.random.random_sample( wshape ) - 1 )
    #         else:                   new_weights.append( np.zeros( wshape, dtype=float ) )
    #     return new_weights

    def getInitialModel(self, fitResult: FitResult ) -> Model:
        model = self.createSequentialModel()
        model.set_weights( fitResult.initial_weights )
        return model

    def getFittedModel( self, fitResult: FitResult ) -> Sequential:
        model = self.createSequentialModel()  # type: Sequential
        model.set_weights( fitResult.final_weights )
#        model.fit( self.inputData, self.outputData, batch_size=self.batchSize, epochs=fitResult.nEpocs, validation_split=self.validation_fraction, shuffle=self.shuffle, verbose=0 )  # type: History
        return model

    # @classmethod
    # def getActivationBackProjection( cls, instance, target_value, iLayer = -1, iHiddenUnit = 0, learning_rate = 0.002 ):
    #     import keras.backend as K
    #     learningModel, result = cls.loadInstance( instance )
    #     model = learningModel.getFittedModel( result )
    #
    #     out_diff = K.mean( (model.layers[-1].output - target_value ) ** 2 ) if iHiddenUnit is None else  K.abs( model.layers[-2].output[0, iHiddenUnit] - target_value )
    #     grad = K.gradients(out_diff, [model.input])[0]
    #     grad /= K.maximum(K.sqrt(K.mean(grad ** 2)), K.epsilon())
    #     iterate = K.function( [model.input, K.learning_phase()], [out_diff, grad] )
    #     input_img_data = np.zeros( shape=model.weights[0].shape[::-1] )
    #     print("Back Projection Map, instance = " + instance + ", Iterations:")
    #     out_loss = 0.0
    #     for i1 in range(5):
    #         for i2 in range(500):
    #             out_loss, out_grad = iterate([input_img_data, 0])
    #             input_img_data -= out_grad * learning_rate
    #             if out_loss < 0.01:
    #                 print("  --> Converged, niters = " + str(i1*500+i2) + ": loss = " + str(out_loss))
    #                 break
    #         if( out_loss > 1.0 ):
    #             learning_rate = learning_rate * 2.0
    #             print("    ** Doubling Learning Rate: niters = " + str( (i1+1) * 500 ) + ": loss = " + str(out_loss))
    #         elif out_loss < 0.01: break
    #
    #     return ( learningModel, model, input_img_data[0] )

    def createSequentialModel( self )-> Sequential:
        model = Sequential()
        nInputs = self.inputData.shape[1]

        if self.layers is not None:
            for iLayer, layer in enumerate(self.layers):
                kwargs = { "input_dim": nInputs } if iLayer == 0 else {}
                instance = layer.instance(**kwargs)
                self.layer_instances.append( instance )
                model.add( instance )

        elif self.hidden is not None:
            nHidden = len(self.hidden)
            nOutputs = 1
            for hIndex in range(nHidden):
                if hIndex == 0:
                    if self.eager:  model.add(Dense(units=self.hidden[hIndex], activation=self.activation, input_tensor=self.inputData))
                    else:           model.add(Dense(units=self.hidden[hIndex], activation=self.activation, input_dim=nInputs))
                else:
                    model.add(Dense(units=self.hidden[hIndex], activation=self.activation))
            output_layer = Dense(units=nOutputs) if nHidden else Dense(units=nOutputs, input_dim=nInputs)
            model.add( output_layer )

        sgd = SGD( lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov )
 #       model.compile(loss=self.lossFunction, optimizer=sgd, metrics=['accuracy'])
        model.compile(loss=self.lossFunction, optimizer='sgd', metrics=['accuracy'])
        if self.weights is not None: model.set_weights(self.weights)
        return model

    # def createEagerSequentialModel( self ):
    #     # type: () -> Sequential
    #     model = Sequential()
    #     nInputs = self.inputs.getInputDimension()
    #     kwargs = {}
    #     model.add( InputLayer( input_tensor=tf.convert_to_tensor(self.inputData, np.float32) ) )
    #
    #     if self.layers is not None:
    #         for iLayer, layer in enumerate(self.layers):
    #             model.add( layer.instance(**kwargs) )
    #
    #     elif self.hidden is not None:
    #         nHidden = len(self.hidden)
    #         nOutputs = self.outputs.getOutputSize()
    #         for hIndex in range(nHidden):
    #             if hIndex == 0:
    #                 model.add(Dense(units=self.hidden[hIndex], activation=self.activation ) )
    #             else:
    #                 model.add(Dense(units=self.hidden[hIndex], activation=self.activation ) )
    #         output_layer = Dense(units=nOutputs) if nHidden else Dense(units=nOutputs )
    #         model.add( output_layer )
    #
    #     sgd = SGD( lr=self.lr, decay=self.decay, momentum=self.momentum, nesterov=self.nesterov )
    #     model.compile(loss=self.lossFunction, optimizer=sgd, metrics=['accuracy'])
    #     if self.weights is not None: model.set_weights(self.weights)
    #     return model

    def updateHistory( self, history: History, initial_weights: List[np.ndarray], iterIndex: int, procIndex: int )-> History:
        if self.performanceTracker.nEpoc > 0:
            if not self.bestFitResult or ( self.performanceTracker.minValLoss < self.bestFitResult.val_loss ):
                self.bestFitResult = FitResult.new( history, initial_weights, self.performanceTracker.getWeights(), self.performanceTracker.minTrainLoss, self.performanceTracker.minValLoss, self.performanceTracker.nEpoc )
            if procIndex < 0:   print("Executed iteration {0}, current loss = {1}  (NE={2}), best loss = {3} (NE={4})".format(iterIndex, self.performanceTracker.minValLoss, self.performanceTracker.nEpoc, self.bestFitResult.val_loss, self.bestFitResult.nEpocs ))
            else:               logging.info( "PROC[{0}]: Iteration {1}, val_loss = {2}, val_loss index = {3}, min val_loss = {4}".format( procIndex, iterIndex, self.performanceTracker.minValLoss, self.performanceTracker.nEpoc, self.bestFitResult.val_loss ) )
        return history

    def plotPrediction( self, fitResult: FitResult, title: str, **kwargs ) -> None:
        print("Plotting result: " + title)
        print(" ---> NEpocs = {0}, val loss = {1}, training loss = {2}".format(fitResult.nEpocs, fitResult.val_loss, fitResult.train_loss ))
        model1: Sequential = self.getFittedModel(fitResult)
        prediction1 = model1.predict( self.inputData )  # type: np.ndarray
        model0: Sequential = self.getInitialModel(fitResult)
        prediction0 = model0.predict( self.inputData )  # type: np.ndarray
        nSteps = prediction1.shape[0]
        for iPlot in range(prediction1.shape[1]):
            plt.title(  "Bathyml: " + title )

            plt.plot(range(nSteps), self.outputData,      "b--", label="training data" )
#            plt.plot(range(nSteps), prediction0[:,iPlot], "g--", label="prediction: init")
            plt.plot(range(nSteps), prediction1[:, iPlot], "r-", label="prediction: fitted")
            plt.legend()
            plt.show()

    def plotPerformance( self, fitResult: FitResult, title: str, **kwargs ):
        valLossHistory = fitResult.valLossHistory()
        trainLossHistory = fitResult.lossHistory()
        plotAverages = kwargs.get("ave",True)
        plt.title(title)
        print("Plotting result: " + title)
        x = list(range( valLossHistory.shape[0]))
        plt.plot( x, valLossHistory, "r-", label="Validation Loss" )
        plt.plot( x, trainLossHistory, "b--", label="Training Loss" )
        if plotAverages and (fitResult.nInstances >0):
            ( aveTrainLossHistory, aveValLossHistory, nI ) = fitResult.getAverages()
            if aveValLossHistory is not None:
                len1 = aveValLossHistory.shape[0]
                plt.plot( x[0:len1], aveValLossHistory, "y-.", label="Average Validation Loss" )
            if aveTrainLossHistory is not None:
                len1 = aveTrainLossHistory.shape[0]
                plt.plot( x[0:len1], aveTrainLossHistory, "y:", label="Average Training Loss" )
            print("Plotting averages for nInstances: " + str(nI))
        plt.legend()
        plt.show()

    @classmethod
    def serial_execute(cls, lmodel_factory, nInterations, shuffle, procIndex=-1, resultQueue=None) -> Union[FitResult,str]:
        ## type: (object(), int, int, int, mp.Queue)
        try:
            learningModel = lmodel_factory()  # type: LearningModel
            result = learningModel.execute( nInterations, shuffle, procIndex )
            if resultQueue is not None: resultQueue.put( result )
            return result
        except Exception as err:
            msg = "PROC[{0}]: {1}".format( procIndex, str(err) )
            logging.error( msg )
            logging.error(traceback.format_exc())
            if resultQueue is not None: resultQueue.put(msg)
            raise err

    @classmethod
    def terminate( cls, processes ):
        print(" ** Terminating worker processes")
        for proc in processes:
            proc.terminate()

    @classmethod
    def fit(cls, learning_model_factory, nIterPerProc, nShuffles, parallel=True, nProc=mp.cpu_count() )-> FitResult:
        ## type: ( object(), int, bool, int)
        if parallel:
            return cls.parallel_execute( learning_model_factory, nIterPerProc, nShuffles, nProc )
        else:
            result =  cls.serial_execute( learning_model_factory, nIterPerProc, nShuffles, nProc )
            if isinstance(result, str): raise Exception( result )
            return result

    @classmethod
    def parallel_execute(cls, learning_model_factory, nIterPerProc, shuffle, nProc=mp.cpu_count() )-> FitResult:
        ## type: ( object(), int, bool, int)
        t0 = time.time()
        result = None
        learning_processes = []
        try:
            results = []
            resultQueue = mp.Queue()

            print(" ** Executing {0} iterations on each of {1} processors for a total of {2} iterations, logging to {3}".format(nIterPerProc,nProc,nIterPerProc*nProc,LOG_FILE))

            for iProc in range(nProc):
                print(" ** Executing process " + str(iProc) + ": NIterations: " + str(nIterPerProc))
                proc = mp.Process(target=cls.serial_execute, args=(learning_model_factory, nIterPerProc, shuffle, iProc, resultQueue))
                proc.start()
                learning_processes.append(proc)

            for iProc in range(mp.cpu_count()):
                print(" ** Waiting for result from process " + str(iProc))
                results.append( resultQueue.get() )

            result = FitResult.getBest(results)

        except Exception:
            traceback.print_exc()

        except KeyboardInterrupt:
            pass

        cls.terminate(learning_processes)
        print("Got Best result, valuation loss = " + str( result.val_loss ) + " training loss = " + str( result.train_loss ) + " in time " + str( time.time() - t0 ) + " secs")
        return result

