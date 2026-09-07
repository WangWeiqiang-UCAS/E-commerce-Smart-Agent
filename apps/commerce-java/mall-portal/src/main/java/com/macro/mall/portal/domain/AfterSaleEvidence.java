package com.macro.mall.portal.domain;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

import java.util.List;

public final class AfterSaleEvidence {
    private AfterSaleEvidence() { }

    public record UploadResponse(String objectKey, String contentType, long contentLength) { }

    public record AttachRequest(
            @NotEmpty @Size(max = 3)
            List<@Pattern(regexp = "^after-sales/[1-9][0-9]*/[0-9a-f-]{36}\\.(jpg|png|pdf)$") String> objectKeys) { }

    public record DownloadResponse(String downloadUrl) { }
}
