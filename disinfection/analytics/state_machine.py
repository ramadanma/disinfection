# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PersonStateMachine:
    """
    Manages the pool-entry state for all tracked persons.

    Operates on the shared `persons` dict (keyed by person_id), modifying it
    in-place.  Each entry is a plain dict with the following keys:

      state             str   'outside' | 'in_pool' | 'qualified' | 'left_early'
      entry_time        float wall-clock seconds when pool entry was confirmed
      total_duration    float seconds spent in pool during the current visit
      stable_count      int   consecutive in-pool frames (debounce entry)
      exit_count        int   consecutive out-of-pool frames (debounce exit)
      should_save_image bool  signal: save the current frame as evidence
      should_record     bool  signal: send a violation report
      has_recorded      bool  prevents duplicate reports for the same visit
      unqualified_count int   total violation count for this track
    """

    def __init__(self, pool_region, stable_threshold: int = 2,
                 required_time: float = 10.0, distance: float = 1.0):
        self.pool_region = pool_region
        self.stable_threshold = stable_threshold
        self.required_time = required_time
        self.distance = distance

    @staticmethod
    def _new_person() -> dict:
        return {
            'state': 'outside',
            'entry_time': None,
            'total_duration': 0.0,
            'stable_count': 0,
            'exit_count': 0,
            'should_save_image': False,
            'should_record': False,
            'has_recorded': False,
            'unqualified_count': 0,
        }

    def update(self, persons: dict, persons_images: dict,
               person_id: int, both_feet_in: bool, current_time: float,
               foot_positions=None, nose_position=None) -> dict:
        """
        Update state for one tracked person; returns that person's dict.

        persons         shared dict  {person_id: person_dict}
        persons_images  shared dict  passed for API compatibility, not used here
        person_id       int          YOLO track id
        both_feet_in    bool         whether both ankles are inside the pool region
        current_time    float        wall-clock timestamp (seconds)
        """
        if person_id not in persons:
            persons[person_id] = self._new_person()

        person = persons[person_id]
        person['should_save_image'] = False  # reset every frame; set below when needed

        if both_feet_in:
            person['exit_count'] = 0
            person['stable_count'] += 1
            state = person['state']

            if state == 'outside':
                if person['stable_count'] >= self.stable_threshold:
                    person['state'] = 'in_pool'
                    person['entry_time'] = current_time
                    person['total_duration'] = 0.0
                    person['has_recorded'] = False
                    logger.info("Person %s entered pool", person_id)

            elif state == 'in_pool':
                person['total_duration'] = current_time - person['entry_time']
                person['should_save_image'] = True
                if person['total_duration'] >= self.required_time:
                    person['state'] = 'qualified'
                    logger.info("Person %s qualified (%.1fs)", person_id, person['total_duration'])

            elif state == 'qualified':
                person['total_duration'] = current_time - person['entry_time']
                person['should_save_image'] = True

            elif state == 'left_early':
                # Re-entered after an early exit; restart timing
                if person['stable_count'] >= self.stable_threshold:
                    person['state'] = 'in_pool'
                    person['entry_time'] = current_time
                    person['total_duration'] = 0.0
                    person['has_recorded'] = False
                    logger.info("Person %s re-entered pool", person_id)

        else:  # both_feet_in is False
            person['stable_count'] = 0
            person['exit_count'] += 1

            if person['exit_count'] < self.stable_threshold:
                # Debounce: ignore a brief out-of-pool reading
                pass

            else:
                state = person['state']
                if state == 'in_pool':
                    person['state'] = 'left_early'
                    person['unqualified_count'] += 1
                    logger.info(
                        "Person %s left pool early (%.1fs / %.1fs required)",
                        person_id, person['total_duration'], self.required_time,
                    )
                    if not person['has_recorded']:
                        person['should_record'] = True

                elif state == 'qualified':
                    person['state'] = 'outside'
                    person['entry_time'] = None
                    logger.info("Person %s exited pool (qualified)", person_id)

        return person
