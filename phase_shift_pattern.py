import cv2
import numpy as np
import matplotlib.pyplot as plt
#from hikvision import HikCamera
import os

class PhaseShiftPattern(object):
    def __init__(self, display_resolution, min_spatial_periods=[32, 32], frequency_multipliers=[4, 4], mask_threshold = 50):
        self.display_resolution = display_resolution
        self.frequency_multipliers = frequency_multipliers
        self.mask_threshold = mask_threshold
        self.periods_x = self.determine_spatial_period(min_spatial_periods[0], frequency_multipliers[0], display_resolution[0])
        self.periods_y = self.determine_spatial_period(min_spatial_periods[1], frequency_multipliers[1], display_resolution[1])

    @staticmethod
    def determine_spatial_period(start, multiplier, threshold):
        numbers = [start]
        while numbers[-1] < threshold:
            numbers.append(numbers[-1] * multiplier)
        return numbers[::-1]

    def generate(self):
        patterns = []

        # Black white
        patterns.append(np.zeros((self.display_resolution[1], self.display_resolution[0]), dtype=np.uint8))
        patterns.append(np.full((self.display_resolution[1], self.display_resolution[0]), 255, dtype=np.uint8))

        # X patterns
        for period in self.periods_x:
            x_patterns = np.zeros( (3, self.display_resolution[1], self.display_resolution[0]), dtype=np.float32)
            for x in range(self.display_resolution[0]):
                x_patterns[0, :, x] = np.cos(2 * np.pi * x /  period - 2 * np.pi / 3)
                x_patterns[1, :, x] = np.cos(2 * np.pi * x /  period)
                x_patterns[2, :, x] = np.cos(2 * np.pi * x /  period + 2 * np.pi / 3)
            patterns.extend(((x_patterns + 1.0) / 2.0 * 255).astype(np.uint8))

        # Y patterns
        for period in self.periods_y:
            y_patterns = np.zeros((3, self.display_resolution[1], self.display_resolution[0]), dtype=np.float32)
            for y in range(self.display_resolution[1]):
                y_patterns[0, y, :] = np.cos(2 * np.pi * y / period - 2 * np.pi / 3)
                y_patterns[1, y, :] = np.cos(2 * np.pi * y / period)
                y_patterns[2, y, :] = np.cos(2 * np.pi * y / period + 2 * np.pi / 3)
            patterns.extend(((y_patterns + 1.0) / 2.0 * 255).astype(np.uint8))

        return patterns

    def decode(self, captures):
        # Find phase limits
        x_phases = self.map_phase_positive(2 * np.pi * np.linspace(0, self.display_resolution[0] - 1, self.display_resolution[0]) / self.periods_x[0])
        x_phases_range = [x_phases.min(), x_phases.max()]
        y_phases = self.map_phase_positive(2 * np.pi * np.linspace(0, self.display_resolution[1] - 1, self.display_resolution[1]) / self.periods_y[0])
        y_phases_range = [y_phases.min(), y_phases.max()]

        # Make mask
        valid_mask = captures[1].astype(np.int16) - captures[0].astype(np.int16) >= self.mask_threshold
        captures = np.array(captures)[2:].astype(np.float32)

        # Find wrapped phase
        wrapped_phases = []
        while 0 < captures.size:
            wrapped_phases.append(self.find_wrapped_phase(captures[:3]))
            captures = captures[3:] # Remove 3 from array
        wrapped_phases_x = wrapped_phases[:len(self.periods_x)]
        wrapped_phases_y = wrapped_phases[len(self.periods_x):]

        # Unwrap phase
        unwrapped_phase_x = self.unwrap_phase(wrapped_phases_x, self.frequency_multipliers[0], x_phases_range)
        unwrapped_phase_y = self.unwrap_phase(wrapped_phases_y, self.frequency_multipliers[1], y_phases_range)

        # Mask invalid pixels as NaN
        unwrapped_phase_x[valid_mask == 0] = np.nan
        unwrapped_phase_y[valid_mask == 0] = np.nan

        # Get decode map my multiplying LCD resolution
        decode_maps = np.stack([unwrapped_phase_x, unwrapped_phase_y], axis=-1)

        print(f'!!!!!{decode_maps.shape}')

        return decode_maps

    @staticmethod
    def map_phase_positive(phases):
        return (phases + 2 * np.pi) % (2 * np.pi)

    @staticmethod
    def find_wrapped_phase(captures):
        return PhaseShiftPattern.map_phase_positive(np.arctan2(np.sqrt(3) * (captures[0] - captures[2]), 2 * captures[1] - captures[0] - captures[2]))

    @staticmethod
    def unwrap_phase(wrapped_phases, frequency_multiplier, phase_range):
        unwrapped_phase = wrapped_phases[0]
        relative_frequency = frequency_multiplier

        for wrapped_phase in wrapped_phases[1:]:
            fringe_order = np.round((unwrapped_phase * relative_frequency - wrapped_phase) / (2 * np.pi))
            unwrapped_phase = (wrapped_phase + fringe_order * 2 * np.pi) / relative_frequency
            relative_frequency *= frequency_multiplier

        # Normalize
        unwrapped_phase = (unwrapped_phase - phase_range[0]) / (phase_range[1] - phase_range[0])
        unwrapped_phase[(unwrapped_phase < 0) | (unwrapped_phase > 1)] = 0
        return unwrapped_phase