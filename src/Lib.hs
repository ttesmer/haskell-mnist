module Lib
    ( train
    ) where

import Data.Matrix as M

train :: Integer -> M.Matrix Double -> M.Matrix Double -> M.Matrix Double -> IO ()
train epoch x y weights = do
    let l1 = sigmoid <$> M.multStd2 x weights
    let l1Error = y - l1
    let l1Delta = arrayMultiply l1Error (sigmoid' <$> l1)
    let updatedWeights = weights + M.multStd2 (M.transpose x) l1Delta
    if epoch == 0
    then print l1
    else train (epoch-1) x y updatedWeights
       where sigmoid  x = 1 / (1 + (exp (-x)))
             sigmoid' x = x * (1 - x)

-- This is NOT matrix multiplication --
arrayMultiply :: Num a => Matrix a -> Matrix a -> Matrix a
arrayMultiply a b = M.fromList 3 1 $ zipWith (*) (M.toList a) (M.toList b)
