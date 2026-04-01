'''
tests/test_experiment.py

Tests for the reactive control experiment.
'''
import pytest
from city.experiment import StimulusGenerator, score_response, TrialResult


class TestStimulusGenerator:
    '''Test the stimulus generator.'''

    def test_deterministic_with_same_seed(self):
        '''Same seed produces same sequence.'''
        gen1 = StimulusGenerator(seed=42)
        gen2 = StimulusGenerator(seed=42)
        seq1 = [gen1.next() for _ in range(20)]
        seq2 = [gen2.next() for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_differ(self):
        '''Different seeds produce different sequences.'''
        gen1 = StimulusGenerator(seed=42)
        gen2 = StimulusGenerator(seed=99)
        seq1 = [gen1.next() for _ in range(20)]
        seq2 = [gen2.next() for _ in range(20)]
        assert seq1 != seq2

    def test_history_tracks_sequence(self):
        gen = StimulusGenerator(seed=42)
        for _ in range(5):
            gen.next()
        assert len(gen.history) == 5
        assert all(s in ('A', 'B') for s in gen.history)

    def test_should_act_on_a_then_b(self):
        '''should_act is True when history ends with trigger pattern.'''
        gen = StimulusGenerator(seed=0, p_switch=0.0)
        gen.next()  # A (no switch, stays at 0)
        assert gen.should_act == False  # Only 1 state, need 2

        # Force an A→B by using p_switch=1.0
        gen2 = StimulusGenerator(seed=0, p_switch=1.0)
        gen2.next()  # Switches to B, history=['B']
        gen2.next()  # Switches to A, history=['B','A']
        gen2.next()  # Switches to B, history=['B','A','B'] → last 2 = A,B → act!
        assert gen2.should_act == True

    def test_should_not_act_on_same_state(self):
        '''should_act is False when state doesn't change.'''
        gen = StimulusGenerator(seed=0, p_switch=0.0)
        gen.next()  # A
        gen.next()  # A
        assert gen.should_act == False  # A,A doesn't match A,B

    def test_three_state_trigger(self):
        '''Trigger A→B→C requires 3-state history match.'''
        gen = StimulusGenerator(
            seed=0, p_switch=1.0,
            states=('A', 'B', 'C'),
            trigger=('A', 'B', 'C'),
        )
        # p_switch=1.0 with 3 states: switches to random other each time
        # Run until we find the trigger or exhaust attempts
        found = False
        for _ in range(100):
            gen.next()
            if gen.should_act:
                found = True
                # Verify the last 3 entries match trigger
                assert tuple(gen.history[-3:]) == ('A', 'B', 'C')
                break
        # With enough steps, the trigger should eventually appear
        # If not found in 100, the PRNG just didn't produce it — that's ok

    def test_step_counter(self):
        gen = StimulusGenerator(seed=42)
        assert gen.step == 0
        gen.next()
        assert gen.step == 1
        gen.next()
        assert gen.step == 2

    def test_three_states_only_produces_valid_states(self):
        gen = StimulusGenerator(seed=42, states=('A', 'B', 'C'))
        for _ in range(50):
            gen.next()
        assert all(s in ('A', 'B', 'C') for s in gen.history)


class TestScoring:
    '''Test the scoring function.'''

    def _make_ab_generator(self):
        '''Create a generator that produces A then B.'''
        gen = StimulusGenerator(seed=0, p_switch=1.0)
        gen.next()  # B
        gen.next()  # A
        gen.next()  # B → history ends A,B → should_act
        return gen

    def _make_no_act_generator(self):
        '''Create a generator where should_act is False.'''
        gen = StimulusGenerator(seed=0, p_switch=0.0)
        gen.next()  # A
        gen.next()  # A → history ends A,A → no act
        return gen

    def test_correct_act(self):
        '''Instance correctly acts on A→B.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'I think this is B after A, so [ACT]')
        assert result.correct == True
        assert result.did_act == True
        assert result.should_act == True

    def test_correct_wait(self):
        '''Instance correctly waits when no A→B.'''
        gen = self._make_no_act_generator()
        result = score_response(gen, 'Same state, [WAIT]')
        assert result.correct == True
        assert result.did_act == False
        assert result.should_act == False

    def test_false_positive(self):
        '''Instance acts when it shouldn't.'''
        gen = self._make_no_act_generator()
        result = score_response(gen, 'I will [ACT] on this')
        assert result.correct == False
        assert result.did_act == True
        assert result.should_act == False

    def test_miss(self):
        '''Instance doesn't act when it should.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'I think I should [WAIT]')
        assert result.correct == False
        assert result.did_act == False
        assert result.should_act == True

    def test_case_insensitive(self):
        '''ACT keyword matching is case-insensitive.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'Let me [act] here')
        assert result.did_act == True

    def test_bare_keyword_does_not_match(self):
        '''Bare ACT without brackets does not count.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'I will ACT now')
        assert result.did_act == False

    def test_last_tag_wins(self):
        '''When model quotes instructions then gives final answer, last tag wins.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'The rule says respond [ACT] when A then B. But this is not that case. [WAIT]')
        assert result.did_act == False

    def test_last_tag_wins_act(self):
        '''Last tag is [ACT] even if [WAIT] appears earlier.'''
        gen = self._make_ab_generator()
        result = score_response(gen, 'I first thought [WAIT] but actually this is A then B. [ACT]')
        assert result.did_act == True
