	</div>
	<div class="spacer"></div>
	<div class="sidebar">
		<div id="fleft">
 <?php if ( !function_exists('dynamic_sidebar') || !dynamic_sidebar("Left Footer") ) : ?>
<h3>Archives</h3>
			<ul>
				<?php wp_get_archives('type=monthly'); ?>
			</ul>
<?php endif; ?>
		</div>
		<div id="fright">
			 <?php if ( !function_exists('dynamic_sidebar') || !dynamic_sidebar("Right Footer") ) : ?>
<h3>Categories</h3>
			<ul>
				<?php wp_list_categories('title_li='); ?>
			</ul>
<?php endif; ?>
		</div>
		<div class="clear"></div>
	</div>