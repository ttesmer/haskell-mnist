module Lib
    ( train
    ) where

import Data.Matrix as M

train :: Integer -> M.Matrix Double -> M.Matrix Double -> M.Matrix Double -> IO ()
train epoch x y weights = do
    let l1 = sigmoid <$> M.multStd2 x weights
    let l1_error = y - l1
    let l1_delta = arrayMultiply l1_error (sigmoid' <$> l1)
    let updatedWeights = weights + M.multStd2 (M.transpose x) l1_delta
    if epoch == 0 then print l1 else train (epoch-1) x y updatedWeights
       where sigmoid' x = x * (1 - x)
             sigmoid  x = 1 / (1 + (exp (-x)))

-- This is NOT matrix multiplication --
arrayMultiply a b = M.fromList 3 1 $ zipWith (*) (M.toList a) (M.toList b)
