{-# LANGUAGE NamedFieldPuns  #-}
{-# LANGUAGE RecordWildCards #-}
module Network
    ( NeuralNet (..),
      train,
      initWB,
    ) where

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Data.List
import           GHC.Base               (build)
import           Numeric.LinearAlgebra  hiding (build, normalize)
import           Prelude                hiding ((<>))
import           Text.Printf

import           Helper

type Bias         = Vector Double
type Weight       = Matrix Double
type Activation   = Matrix Double
type Nabla        = Matrix Double
type ImagesMat    = Matrix Double
type LabelsMat    = Matrix Double

data NeuralNet = NN
    { weights     :: ![Weight]  -- ^ List of all weights
    , biases      :: ![Bias]    -- ^ List of all biases
    , eta         :: !Double    -- ^ Learning rate
    , lambda      :: !Double    -- ^ L2 regularization parameter
    , totalEpochs :: !Int
    , epochs      :: !Int       -- ^ No. of epochs
    , layers      :: !Int       -- ^ No. of layers
    , layerSize   :: !Int       -- ^ Size of hidden layers
    , batchSize   :: !Int       -- ^ Size of mini-batch
    , trainData   :: ![(Image, Label)]
    , testData    :: ![(ImagesMat, Int)]
    } deriving (Show, Eq)

test :: NeuralNet -> IO ()
test NN{..} = do
    let (imgs, labels) = unzip testData
    let guesses = [guess img (zip weights $ toMatrix <$> biases) | img <- imgs]
    let correct = sum $ (\(a,b) -> if a == b then 1 else 0) <$!!> zip guesses labels
    printf "Epoch %d: %.2f%%\n" (totalEpochs-epochs :: Int)  (correct/100 :: Float)
    where
       guess :: ImagesMat -> [(Weight, Matrix Double)] -> Int
       guess i wb = maxIndex . flatten . head $ feedforward i wb

train :: NeuralNet -> IO NeuralNet
train net@NN{..} = do
    shuffledData <- shuffle trainData
    let miniBatches = getMiniBatch <$!!> chunkList batchSize shuffledData
    let newNet = foldl' trainBatch net miniBatches
    test newNet
    case epochs of
        0 -> return newNet
        _ -> train newNet { epochs=epochs-1 }

trainBatch :: NeuralNet -> (ImagesMat, LabelsMat) -> NeuralNet
trainBatch net@NN{weights=ws, biases=bs, ..} (mX, mY) = newNet
    where newNet = let activations = feedforward mX (zip ws $ bTm bs batchSize)
                      in updateMiniBatch net $
                          backprop mY net activations

-- | Make biases suitable for full-matrix backprop.
bTm :: [Bias] -> Int -> [Matrix Double]
bTm biases batchSize = fmap (fromColumns . replicate batchSize) biases

-- | Turn mini-batch matrices into one matrix.
getMiniBatch :: [(Image, Label)] -> (ImagesMat, LabelsMat)
getMiniBatch mBatchData = (fromColumns imgs, fromColumns labels)
  where
    (imgs, labels) = unzip mBatchData

--  | Returns activations, given input image(s) 
--  | and neural network
feedforward :: ImagesMat -> [(Weight, Matrix Double)] -> [Activation]
feedforward = scanr (\(w, b) a -> sigmoid $ w <> a + b)

--  | Updates mini-batch
--  NOTE: use 'seq' or 'deepseq' for full evaluation.
updateMiniBatch :: NeuralNet -> ([Vector Double], [Matrix Double])  -> NeuralNet
updateMiniBatch net@NN{..} (nablaB, nablaW) = net {weights=wnew, biases=bnew}
    where
        wnew :: [Weight]
        wnew = force [ scale (1-eta*lambda/n) w -
                   scale (eta/fromIntegral batchSize) nw |
            (w, nw)  <- zip weights nablaW ]
        bnew :: [Bias]
        bnew = [ b - scale (eta/fromIntegral batchSize) nb |
            (b, nb)  <- zip biases nablaB ]
        n = fromIntegral $ length trainData

--  | Returns nablaW and nablaB
backprop :: LabelsMat -> NeuralNet -> [Activation] -> ([Vector Double], [Matrix Double])
backprop y NN{..} (aL:as) = nablaW `deepseq` (nablaB, nablaW)
    where
        nablaW :: [Matrix Double]
        nablaW = zipWith (<>) delta $ tr <$!!> as
        nablaB :: [Vector Double]
        nablaB = sumRows <$!!> delta
        delta :: [Matrix Double]
        delta = init $ deltas [crossEntropy y aL] y as weights

deltas :: [Matrix Double] -> LabelsMat -> [Activation] -> [Weight] -> [Matrix Double]
deltas deltal _ [] [] = deltal
deltas deltal y (a:as) (wlp1:ws) = deltas (deltal++[(tr wlp1 <> last deltal) * sigmoid' a]) y as ws

quadraticLoss :: LabelsMat -> Activation -> Matrix Double
quadraticLoss y aL = (aL - y) * sigmoid' aL

crossEntropy :: LabelsMat -> Activation -> Matrix Double
crossEntropy y aL = aL - y

{-- Arithmetic Functions --}
sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Floating a => a -> a
sigmoid' x = x * (1 - x)

{-- Data Processing Functions --}
initWB :: NeuralNet -> IO NeuralNet
initWB net@NN{batchSize = bs, layerSize = ls, layers = n} = do
    headW <- randn ls 784
    midW  <- replicateM (n-2) (randn ls ls)
    lastW <- randn 10 ls

    initB <- replicateM (n-1) (randn ls 1)
    lastB <- randn 10 1

    -- divBy is used for sharper peak in Gaussian,
    -- see here: http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
    let weights = force . reverse $ divBy (sqrt 784) headW : map (divBy (sqrt $ fromIntegral ls))  midW ++ [divBy (fromIntegral ls) lastW]
    let biases = force . reverse . map flatten $ initB ++ [lastB]

    return net {weights=weights, biases=biases}

