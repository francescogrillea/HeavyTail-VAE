classdef reshapeLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable

    properties
        % Layer properties.
        OutputSize
    end

    methods
        function layer = reshapeLayer(outputSize, NameValueArgs)
            % layer = reshapeLayer(outputSize)
            % creates a reshapeLayer object that
            % reshapes the input to the specified output size.

            % Parse input arguments.
            arguments
                outputSize
                NameValueArgs.Name = "";
            end

            % Set layer name.
            name = NameValueArgs.Name;
            layer.Name = name;

            % Set layer description.
            layer.Description = "Reshape to size " + ...
                join(string(outputSize));

            % Set layer type.
            layer.Type = "Reshape";

            % Set output size.
            layer.OutputSize = outputSize;
        end

        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data, specified as a formatted dlarray
            %                 with a "C" and optionally a "B" dimension.
            % Outputs:
            %         Z     - Output of layer forward function returned as
            %                 a formatted dlarray with format "SSCB".

            % Reshape.
            outputSize = layer.OutputSize;
            Z = reshape(X, outputSize(1), outputSize(2), outputSize(3),[]);
            Z = dlarray(Z, "SSCB");
        end
    end
end