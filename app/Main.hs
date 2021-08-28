module Main where
import Lib
import qualified Data.Matrix as M

main :: IO ()
main = do
    -- has to be n^2 matrix
    let x = M.fromLists [[1,0,1,1,0,1],
                         [0,1,0,0,1,0],
                         [0,1,1,1,0,0],
                         [1,0,1,0,0,1],
                         [1,1,1,0,0,1],
                         [0,0,1,1,0,1]] 

    -- dynamically get the first column as goal matrix
    let y = M.colVector $ M.getCol 1 x

    putStrLn "Training Data X:"
    print x
    putStrLn "Desired Ouput Y:"
    print y

    -- weight matrix with size according to number of columns in base matrix
    syn0 <- weightMatrix $ M.ncols x

    putStrLn "Result After Training:"
    train 60000 x y syn0
