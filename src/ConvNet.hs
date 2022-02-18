{-# LANGUAGE NamedFieldPuns  #-}
{-# LANGUAGE RecordWildCards #-}
module ConvNet
    ( CNN (..)
    , initKernels
    , train
    ) where

import           Control.DeepSeq
import           Control.Lens
import           Control.Monad
import           Data.List
import           Numeric.LinearAlgebra
import           Prelude               hiding ((<>))
import           Text.Printf

import           Helper                hiding (Image)

type Bias         = Matrix Double
type Weight       = Matrix Double
type Kernel       = Matrix Double
type FeatureMap   = Matrix Double
type Activation   = Matrix Double
type Image        = Matrix Double

--  | Convolutional Neural Network
data CNN = CNN
    { kernels        :: ![Matrix Double]   --  ^ List of all kernels (shared weights)
    , kernelSize     :: !(Int, Int, Int) --  ^ Size of kernels
    , batchSize      :: !Int        --  ^ Size of mini-batches 
    , trainData   :: ![(Image, Label)]
    } deriving (Show, Eq)

train :: CNN -> [[[FeatureMap]]]
train net@CNN{..} = (convolve net) <$> miniBatches
    where miniBatches = chunkList batchSize trainData

--  | TODO: parallelize convolutions
convolve :: CNN -> [(Image, Label)] -> [[FeatureMap]]
convolve net@CNN{..} mBatch = [ map (sigmoid . conv2 k) imgs | k <- kernels]
    where (imgs, labels) = unzip mBatch

--  | Initializes shared weights of feature maps
initKernels :: CNN -> IO CNN
initKernels net@CNN{kernelSize=(n,r,c), ..} = do
    kernels <- replicateM n $ randn r c
    return net { kernels=kernels }

{-- Arithmetic Functions --}
softmax :: Matrix Double  -> Matrix Double
softmax z = expZ / (scalar $ sumElements expZ)
    where expZ = cmap exp z

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Floating a => a -> a
sigmoid' x = x * (1 - x)

