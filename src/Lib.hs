module Lib
    ( train,
      weightMatrix
    ) where

import qualified Data.Matrix as M
import qualified System.Random as R
import qualified Control.Monad as C

weightMatrix :: Int -> IO (M.Matrix Double)
weightMatrix col = (C.replicateM col $ (R.randomRIO (0 :: Double, 1 :: Double))) >>= (\x -> return $ M.fromList col 1 ((\y -> 2*y -1) <$> x))

train :: Integer -> M.Matrix Double -> M.Matrix Double -> M.Matrix Double -> IO ()
train epoch x y weights = do
    let l1 = sigmoid <$> M.multStd2 x weights
    let l1Error = y - l1
    let l1Delta = M.elementwise (*) l1Error $ sigmoid' <$> l1
    let updatedWeights = weights + M.multStd2 (M.transpose x) l1Delta
    if epoch == 0
    then print l1
    else train (epoch-1) x y updatedWeights
       where sigmoid  x = 1 / (1 + (exp (-x))) -- sigmoid activation function
             sigmoid' x = x * (1 - x)
