import unittest
from unittest.mock import patch

import tensors as ts
from tensors.graph import Computation
from tensors.graph import computation as computation_module
from tensors.graph import fusion
from tensors.graph.state import reset_graph_state


class FusionPlanningTests(unittest.TestCase):
    """Fusion is planned from the instruction sequence, beside it."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_computation_delegates_planning_to_the_fusion_module(self):
        value = ts.Variable(ts.full((4_096,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)
        computation = Computation(output)

        # Nothing but the instructions and their Variables decides a fusion,
        # so planning them again outside the Computation reproduces the plan.
        fusions, fusion_starts = fusion.plan_fusions(
            computation._instructions,
            computation._variables,
        )

        self.assertEqual(fusions, computation._fusions)
        self.assertEqual(fusion_starts, computation._fusion_starts)

    def test_plan_fusions_describes_each_step_of_a_chain(self):
        left = ts.Variable(ts.full((64,), 0.5))
        right = ts.Variable(ts.full((64,), 1.5))
        output = ts.sum(right / (ts.sin(left * 2.0) + 0.25))
        computation = Computation(output)

        self.assertEqual(
            [instruction.operation.name
             for instruction in computation._instructions],
            ["mul", "sin", "add", "div", "sum"],
        )
        # The four elementwise instructions fuse; the reduction stays
        # ordinary. The final step divides by the chain, so it records the
        # reversed operand order.
        self.assertEqual(list(computation._fusions), [0])
        end, steps, source_slots = computation._fusions[0]
        self.assertEqual(end, 3)
        self.assertEqual(
            steps,
            (
                ("multiply", None, False, 1),
                ("sin", None, False, None),
                ("add", None, False, 2),
                ("divide", None, True, 3),
            ),
        )
        sourced = {computation._variables[slot] for slot in source_slots}
        self.assertEqual(len(source_slots), 4)
        self.assertTrue({left, right} <= sourced)

    def test_fusion_starts_index_runs_by_their_final_instruction(self):
        value = ts.Variable(ts.full((64,), 0.5))
        output = ts.sum(ts.sin(value * 1.5) + 0.25)
        computation = Computation(output)

        starts = {
            end: start
            for start, (end, _, _) in computation._fusions.items()
        }
        self.assertEqual(computation._fusion_starts, starts)

    def test_fused_operation_names_only_fusible_invocations(self):
        value = ts.Variable(ts.full((64,), 0.5))
        output = ts.sum(ts.sin(value * 1.5))
        instructions = Computation(output)._instructions

        self.assertEqual(
            [fusion.fused_operation(instruction)
             for instruction in instructions],
            ["multiply", "sin", None],
        )


class FusionModuleBoundaryTests(unittest.TestCase):
    """The fusion mechanics live outside the execution model."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    def test_fusion_mechanics_left_the_computation_class(self):
        for name in (
            "_plan_fusions",
            "_fused_operation",
            "_start_fused_instruction",
            "_extend_fused_instruction",
            "_execute_fused_range",
            "_execute_fused_backward_range",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(Computation, name))

    def test_ordinary_instruction_execution_stays_in_the_computation(self):
        for name in ("_execute_instruction", "_execute_backward_instruction"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(Computation, name))

    def test_fusion_module_defines_no_classes(self):
        classes = [
            name
            for name, value in vars(fusion).items()
            if isinstance(value, type) and value.__module__ == fusion.__name__
        ]
        self.assertEqual(classes, [])

    def test_graph_package_does_not_export_fusion_helpers(self):
        helpers = (
            "Fusion",
            "FusedStep",
            "fused_operation",
            "start_fusion",
            "extend_fusion",
            "plan_fusions",
            "execute_fused_forward",
            "execute_fused_backward",
        )
        for name in helpers:
            with self.subTest(name=name):
                self.assertNotIn(name, ts.graph.__all__)
                self.assertFalse(hasattr(ts.graph, name))
                self.assertFalse(hasattr(ts, name))

    def test_fusion_metadata_is_absent_from_the_instruction_type(self):
        for name in ("fused", "fusion", "fusion_id", "fused_steps"):
            with self.subTest(field=name):
                self.assertNotIn(name, computation_module.Instruction.__slots__)


class DeclinedFusionTests(unittest.TestCase):
    """Declining a fusion changes nothing but the path taken."""

    def setUp(self):
        reset_graph_state()

    def tearDown(self):
        reset_graph_state()

    @staticmethod
    def _build():
        value = ts.Variable(ts.full((512,), 0.5), requires_grad=True)
        scale = ts.Variable(ts.full((512,), 1.5), requires_grad=True)
        output = ts.sum(ts.tanh(ts.sin(value * scale) + 0.25))
        return output, value, scale

    def _replay_and_differentiate(self):
        output, value, scale = self._build()
        computation = Computation(output)
        replayed = computation.forward().tolist()
        computation.backward()
        return replayed, value.grad.tolist(), scale.grad.tolist()

    def test_forward_matches_when_the_fused_kernel_declines(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                reset_graph_state()
                expected = self._replay_and_differentiate()

                reset_graph_state()
                with patch.object(
                    computation_module,
                    "execute_fused_forward",
                    return_value=False,
                ) as declined:
                    unfused = self._replay_and_differentiate()

                declined.assert_called()
                self.assertEqual(unfused, expected)

    def test_backward_matches_when_the_fused_vjp_declines(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                reset_graph_state()
                expected = self._replay_and_differentiate()

                reset_graph_state()
                with patch.object(
                    computation_module,
                    "execute_fused_backward",
                    return_value=False,
                ) as declined:
                    unfused = self._replay_and_differentiate()

                declined.assert_called()
                self.assertEqual(unfused, expected)

    def test_grad_matches_with_neither_direction_fused(self):
        for backend in ts.available_backends():
            with self.subTest(backend=backend), ts.use_backend(backend):
                reset_graph_state()
                output, value, scale = self._build()
                expected = [
                    gradient.tolist()
                    for gradient in ts.grad(output, (value, scale))
                ]

                reset_graph_state()
                with (
                    patch.object(
                        computation_module,
                        "execute_fused_forward",
                        return_value=False,
                    ),
                    patch.object(
                        computation_module,
                        "execute_fused_backward",
                        return_value=False,
                    ),
                ):
                    output, value, scale = self._build()
                    Computation(output).forward()
                    unfused = [
                        gradient.tolist()
                        for gradient in ts.grad(output, (value, scale))
                    ]

                self.assertEqual(unfused, expected)


if __name__ == "__main__":
    unittest.main()
