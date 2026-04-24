import unittest
from utgc.chain import CouponCollectorChain, create_partition_iterator

class DummyPartition:
    def __init__(self, value):
        self.value = value
        self.parent = None
        
    def __eq__(self, other):
        if isinstance(other, DummyPartition):
            return self.value == other.value
        return self.value == other

    def __add__(self, other):
        return DummyPartition(self.value + other)


class TestCouponCollectorChain(unittest.TestCase):
    def setUp(self):
        self.initial_state = DummyPartition(0)
        # Proposal increments the state by 1. 
        # This acts as our counter for how many micro-steps were executed!
        self.proposal = lambda state: state + 1
        self.constraints = [lambda state: True]
        self.accept = lambda state: True

    def test_coupon_collector_length(self):
        chain = CouponCollectorChain(
            proposal=self.proposal,
            constraints=self.constraints,
            accept=self.accept,
            initial_state=self.initial_state,
            micro_steps_per_yield=5,
            num_macro_steps=10
        )
        self.assertEqual(len(chain), 10)

    def test_coupon_collector_iteration_count(self):
        chain = CouponCollectorChain(
            proposal=self.proposal,
            constraints=self.constraints,
            accept=self.accept,
            initial_state=self.initial_state,
            micro_steps_per_yield=5,
            num_macro_steps=10
        )
        items = list(chain)
        self.assertEqual(len(items), 10)

    def test_coupon_collector_micro_steps(self):
        chain = CouponCollectorChain(
            proposal=self.proposal,
            constraints=self.constraints,
            accept=self.accept,
            initial_state=self.initial_state,
            micro_steps_per_yield=5,
            num_macro_steps=10
        )
        items = list(chain)
        # 0th iteration: returns initial state (0)
        # 1st iteration: returns state after 5 proposals (0 + 5 = 5)
        # 2nd iteration: returns state after 5 more proposals (5 + 5 = 10)
        # ...
        # 9th iteration (10th item): returns state after 5 * 9 = 45 micro-steps total (45)
        self.assertEqual(items[0], 0)
        self.assertEqual(items[1], 5)
        self.assertEqual(items[-1], 45)

    def test_coupon_collector_0th_iteration(self):
        chain = CouponCollectorChain(
            proposal=self.proposal,
            constraints=self.constraints,
            accept=self.accept,
            initial_state=100,
            micro_steps_per_yield=5,
            num_macro_steps=2
        )
        iterator = iter(chain)
        first_item = next(iterator)
        self.assertEqual(first_item, 100)
        
    def test_create_partition_iterator_coupon_collector(self):
        iterator = create_partition_iterator(
            proposal=self.proposal,
            initial_partition=self.initial_state,
            constraints=self.constraints,
            optimization_scheme_params={
                "scheme": "coupon_collector",
                "micro_steps_per_yield": 5,
                "num_macro_steps": 10
            },
            num_steps=0 # ignored for coupon collector, maybe should be optional?
        )
        # The iterator is wrapped in tqdm by chain.with_progress_bar()
        self.assertEqual(type(iterator.iterable).__name__, "CouponCollectorChain")

if __name__ == '__main__':
    unittest.main()
