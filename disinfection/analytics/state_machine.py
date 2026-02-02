# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class PersonState(Enum):
    UNKNOWN = auto()
    KNOWN = auto()
    STRANGER = auto()
    LOST = auto()


@dataclass
class PersonStateMachine:
    """
        There is one state machine instance for each "person/track." 
        You need to manage multiple instances at the higher level using track_id or face_id.
    """
    state: PersonState = PersonState.UNKNOWN
    person_id: Optional[str] = None  # 例如员工号/库里的人ID

    def update(self, *, recognized_id: Optional[str], visible: bool) -> PersonState:
        """
        recognized_id: 识别到的身份（识别不到则 None）
        visible: 当前帧是否还看到这个人/track
        """
        if not visible:
            self.state = PersonState.LOST
            self.person_id = None
            return self.state

        if recognized_id:
            self.state = PersonState.KNOWN
            self.person_id = recognized_id
        else:
            # 看到了人但识别不到
            if self.state != PersonState.KNOWN:
                self.state = PersonState.STRANGER
                self.person_id = None
        return self.state
