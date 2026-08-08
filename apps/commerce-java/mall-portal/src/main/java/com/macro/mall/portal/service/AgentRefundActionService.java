package com.macro.mall.portal.service;

import com.macro.mall.portal.domain.AgentRefundAction;

public interface AgentRefundActionService {
    AgentRefundAction.PreviewResponse preview(AgentRefundAction.PreviewRequest request);

    AgentRefundAction.ConfirmResponse confirm(AgentRefundAction.ConfirmRequest request);
}
