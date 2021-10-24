module Main where
import           Network
import           System.IO

netConfig :: IO NeuralNet
netConfig = initWB NN
    { weights   = []
    , biases    = []
    , eta       = 3.0
    , epochs    = 30
    , layers    = 2
    , layerSize = 30
    , batchSize = 10
    }

main :: IO ()
main = do
    net <- netConfig
    result <- train net
    print result
