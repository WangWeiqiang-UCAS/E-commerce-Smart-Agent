package com.macro.mall.portal.service;

import com.macro.mall.portal.domain.AgentOrderCancellationAction;

public interface AgentOrderCancellationActionService {
    AgentOrderCancellationAction.PreviewResponse preview(AgentOrderCancellationAction.PreviewRequest request);

    AgentOrderCancellationAction.ConfirmResponse confirm(AgentOrderCancellationAction.ConfirmRequest request);
}
