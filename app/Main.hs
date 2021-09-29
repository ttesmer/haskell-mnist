module Main where
import Network
import System.IO

main :: IO ()
main = do
    syn0 <- randomMatrix 30 784
    syn1 <- randomMatrix 10 30
    bias0 <- randomMatrix 30 1
    bias1 <- randomMatrix 10 1
    --eta <- readDouble
    trainingData <- loadData
    testData <- loadTestData
    learnedWaB <- train 30 10 3.0 [(syn1, bias1), (syn0, bias0)] trainingData testData
    return ()

readDouble :: IO Double
readDouble = readLn
