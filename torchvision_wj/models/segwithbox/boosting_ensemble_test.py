# from enum import Enum, auto
# from typing import Type


# class ModelType(Enum):
#     """
#     Enumeration for different segmentation model types
#     """

#     UNET = auto()
#     RESNET = auto()
#     DEEPLABV3 = auto()

#     def __str__(self):
#         """String representation of the enum"""
#         return self.name.lower()

#     @classmethod
#     def from_string(cls, model_string: str) -> "ModelType":
#         """
#         Convert string to ModelType enum

#         Args:
#             model_string: String representation of model type

#         Returns:
#             ModelType enum value

#         Raises:
#             ValueError: If string doesn't match any model type
#         """
#         model_string = model_string.upper().strip()
#         try:
#             return cls[model_string]
#         except KeyError:
#             raise ValueError(
#                 f"Unknown model type: {model_string}. Available types: {list(cls.__members__.keys())}"
#             )


# class ModelArchitecture(Enum):
#     """
#     Enumeration for model architectures with additional metadata
#     """

#     UNET = ("unet", "U-Net", "Encoder-decoder with skip connections")
#     RESNET = ("resnet", "ResNet-Based", "Residual network with backbone")
#     DEEPLABV3 = ("deeplabv3", "DeepLabV3", "Atrous spatial pyramid pooling")

#     def __init__(self, short_name: str, display_name: str, description: str):
#         self.short_name = short_name
#         self.display_name = display_name
#         self.description = description

#     def __str__(self):
#         return self.display_name

#     @classmethod
#     def get_by_short_name(cls, short_name: str) -> "ModelArchitecture":
#         """
#         Get model architecture by short name

#         Args:
#             short_name: Short name of the model

#         Returns:
#             ModelArchitecture enum value
#         """
#         short_name = short_name.lower().strip()
#         for model in cls:
#             if model.short_name == short_name:
#                 return model
#         raise ValueError(f"Unknown model short name: {short_name}")

#     @property
#     def info(self) -> dict:
#         """Get model information as dictionary"""
#         return {
#             "name": self.name,
#             "short_name": self.short_name,
#             "display_name": self.display_name,
#             "description": self.description,
#         }


# class BoostingConfiguration(Enum):
#     """
#     Enumeration for boosting ensemble configurations
#     """

#     UNET_DEEPLAB_RESNET = (
#         1,
#         "UNet → DeepLabV3 → ResNet",
#         "Detailed → Context → Semantics",
#     )
#     RESNET_UNET_DEEPLAB = (
#         2,
#         "ResNet → UNet → DeepLabV3",
#         "Global → Detail → Multi-scale",
#     )
#     DEEPLAB_UNET_RESNET = (
#         3,
#         "DeepLabV3 → UNet → ResNet",
#         "Multi-scale → Detail → Validation",
#     )

#     def __init__(self, config_id: int, description: str, pipeline: str):
#         self.config_id = config_id
#         self.description = description
#         self.pipeline = pipeline

#     def __str__(self):
#         return f"Configuration {self.config_id}: {self.description}"

#     @classmethod
#     def get_by_id(cls, config_id: int) -> "BoostingConfiguration":
#         """
#         Get configuration by ID

#         Args:
#             config_id: Configuration ID (1, 2, or 3)

#         Returns:
#             BoostingConfiguration enum value
#         """
#         for config in cls:
#             if config.config_id == config_id:
#                 return config
#         raise ValueError(
#             f"Invalid configuration ID: {config_id}. Available: {[c.config_id for c in cls]}"
#         )

#     @property
#     def model_sequence(self) -> list:
#         """Get the model sequence for this configuration"""
#         sequences = {
#             1: [ModelType.UNET, ModelType.DEEPLABV3, ModelType.RESNET],
#             2: [ModelType.RESNET, ModelType.UNET, ModelType.DEEPLABV3],
#             3: [ModelType.DEEPLABV3, ModelType.UNET, ModelType.RESNET],
#         }
#         return sequences[self.config_id]


# class ModelParameters(Enum):
#     """
#     Enumeration for model parameter presets
#     """

#     UNET_LIGHT = (ModelType.UNET, 16, "Lightweight UNet")
#     UNET_STANDARD = (ModelType.UNET, 32, "Standard UNet")
#     UNET_HEAVY = (ModelType.UNET, 64, "Heavy UNet")

#     RESNET_LIGHT = (ModelType.RESNET, "resnet18", "Lightweight ResNet")
#     RESNET_STANDARD = (ModelType.RESNET, "resnet34", "Standard ResNet")
#     RESNET_HEAVY = (ModelType.RESNET, "resnet50", "Heavy ResNet")

#     DEEPLAB_LIGHT = (ModelType.DEEPLABV3, 32, "Lightweight DeepLabV3")
#     DEEPLAB_STANDARD = (ModelType.DEEPLABV3, 64, "Standard DeepLabV3")
#     DEEPLAB_HEAVY = (ModelType.DEEPLABV3, 128, "Heavy DeepLabV3")

#     def __init__(self, model_type: ModelType, param_value, description: str):
#         self.model_type = model_type
#         self.param_value = param_value
#         self.description = description

#     @classmethod
#     def get_presets(cls, model_type: ModelType) -> list["ModelParameters"]:
#         """
#         Get all presets for a specific model type

#         Args:
#             model_type: Type of model

#         Returns:
#             List of model parameter presets
#         """
#         return [preset for preset in cls if preset.model_type == model_type]


# # Usage examples and utility functions
# def print_model_info():
#     """Print information about all available models"""
#     print("=" * 60)
#     print("AVAILABLE SEGMENTATION MODELS")
#     print("=" * 60)

#     for model in ModelArchitecture:
#         print(f"\n{model.display_name} ({model.short_name}):")
#         print(f"  Description: {model.description}")
#         print(f"  Enum name: {model.name}")


# def print_boosting_configurations():
#     """Print information about boosting configurations"""
#     print("\n" + "=" * 60)
#     print("BOOSTING CONFIGURATIONS")
#     print("=" + "=" * 59)

#     for config in BoostingConfiguration:
#         print(f"\n{config}:")
#         print(f"  Pipeline: {config.pipeline}")
#         print(
#             f"  Model sequence: {' → '.join([str(mt) for mt in config.model_sequence])}"
#         )


# def create_model_config(model_type: ModelType, preset: str = "standard") -> dict:
#     """
#     Create model configuration based on type and preset

#     Args:
#         model_type: Type of model to create
#         preset: Model preset ('light', 'standard', 'heavy')

#     Returns:
#         Dictionary with model configuration
#     """
#     preset = preset.upper()
#     preset_name = f"{model_type.name}_{preset}"

#     try:
#         model_params = ModelParameters[preset_name]
#         config = {
#             "model_type": model_type,
#             "preset": preset,
#             "param_value": model_params.param_value,
#             "description": model_params.description,
#         }

#         # Add type-specific parameters
#         if model_type == ModelType.UNET:
#             config["channels"] = model_params.param_value
#         elif model_type == ModelType.RESNET:
#             config["backbone"] = model_params.param_value
#         elif model_type == ModelType.DEEPLABV3:
#             config["channels"] = model_params.param_value

#         return config

#     except KeyError:
#         available_presets = [
#             p.name.split("_")[1] for p in ModelParameters if p.model_type == model_type
#         ]
#         raise ValueError(
#             f"Unknown preset '{preset}' for {model_type}. Available: {available_presets}"
#         )


# def validate_boosting_sequence(sequence: list[ModelType]) -> bool:
#     """
#     Validate if a model sequence is a valid boosting configuration

#     Args:
#         sequence: List of ModelType in sequence

#     Returns:
#         True if valid, False otherwise
#     """
#     valid_sequences = [config.model_sequence for config in BoostingConfiguration]
#     return sequence in valid_sequences


# # In your boosting factory function
# def create_boosting_ensemble_with_enum(
#     model1_type: ModelType,
#     model2_type: ModelType,
#     model3_type: ModelType,
#     configuration: BoostingConfiguration,
#     in_dim: int,
#     out_dim: int,
#     softmax: bool
# ):
#     # Validate sequence
#     sequence = [model1_type, model2_type, model3_type]
#     if not validate_boosting_sequence(sequence):
#         raise ValueError(f"Invalid model sequence for boosting")

#     # Create models based on types and presets
#     model1_config = create_model_config(model1_type, "standard")
#     model2_config = create_model_config(model2_type, "standard")
#     model3_config = create_model_config(model3_type, "standard")

#     # ... rest of implementation

# # Example usage
# config = BoostingConfiguration.UNET_DEEPLAB_RESNET
# model = create_boosting_ensemble_with_enum(
#     ModelType.UNET,
#     ModelType.DEEPLABV3,
#     ModelType.RESNET,
#     config,
#     in_dim=1,
#     out_dim=2,
#     softmax=True
# )

# # Example usage and testing
# if __name__ == "__main__":
#     # Demonstrate enum usage
#     print("=== MODEL TYPE ENUM DEMONSTRATION ===")

#     # Basic enum usage
#     model = ModelType.UNET
#     print(f"\nBasic enum usage:")
#     print(f"Model: {model}")
#     print(f"Name: {model.name}")
#     print(f"Value: {model.value}")
#     print(f"String: {str(model)}")

#     # String conversion
#     print(f"\nString conversion:")
#     model_from_string = ModelType.from_string("resnet")
#     print(f"From string 'resnet': {model_from_string}")

#     # Model architecture with metadata
#     print(f"\nModel architecture info:")
#     arch = ModelArchitecture.DEEPLABV3
#     print(f"Architecture: {arch}")
#     print(f"Short name: {arch.short_name}")
#     print(f"Description: {arch.description}")
#     print(f"Info dict: {arch.info}")

#     # Boosting configurations
#     print(f"\nBoosting configurations:")
#     config = BoostingConfiguration.RESNET_UNET_DEEPLAB
#     print(f"Config: {config}")
#     print(f"ID: {config.config_id}")
#     print(f"Pipeline: {config.pipeline}")
#     print(f"Model sequence: {[str(m) for m in config.model_sequence]}")

#     # Model parameters
#     print(f"\nModel parameters:")
#     params = ModelParameters.UNET_STANDARD
#     print(f"Preset: {params}")
#     print(f"Model type: {params.model_type}")
#     print(f"Param value: {params.param_value}")
#     print(f"Description: {params.description}")

#     # Get all presets for a model type
#     print(f"\nAll UNet presets:")
#     for preset in ModelParameters.get_presets(ModelType.UNET):
#         print(
#             f"  - {preset.name}: {preset.description} (channels: {preset.param_value})"
#         )

#     # Create model configuration
#     print(f"\nModel configuration creation:")
#     config = create_model_config(ModelType.RESNET, "heavy")
#     print(f"ResNet heavy config: {config}")

#     # Validation
#     print(f"\nSequence validation:")
#     valid_sequence = [ModelType.UNET, ModelType.DEEPLABV3, ModelType.RESNET]
#     invalid_sequence = [ModelType.UNET, ModelType.UNET, ModelType.RESNET]

#     print(
#         f"Sequence {[str(m) for m in valid_sequence]}: valid = {validate_boosting_sequence(valid_sequence)}"
#     )
#     print(
#         f"Sequence {[str(m) for m in invalid_sequence]}: valid = {validate_boosting_sequence(invalid_sequence)}"
#     )

#     # Print comprehensive info
#     print_model_info()
#     print_boosting_configurations()

#     # Iterate through all model types
#     print(f"\nAll available model types:")
#     for model_type in ModelType:
#         print(f"  - {model_type.name}: {model_type.value}")

#     # Example of using enums in function parameters
#     def create_model_selection_menu():
#         """Example function using enums for model selection"""
#         print(f"\n=== MODEL SELECTION MENU ===")
#         for i, model in enumerate(ModelArchitecture, 1):
#             print(f"{i}. {model.display_name} - {model.description}")

#         choice = 1  # This would be user input in real application
#         selected_model = list(ModelArchitecture)[choice - 1]
#         print(f"Selected: {selected_model.display_name}")

#         # Show available presets for selected model
#         print(f"Available presets for {selected_model.display_name}:")
#         model_type = ModelType[selected_model.name]
#         for preset in ModelParameters.get_presets(model_type):
#             print(f"  - {preset.name.split('_')[1]}: {preset.description}")

#     create_model_selection_menu()
